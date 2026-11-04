# mixing.probe

One light media probe: `ffprobe` first, ffmpeg’s banner as the fallback.

What a file holds (its displayed size, exact frame rate, frame count, stream
start times, rotation, duration, sound format), read by subprocess so that
importing this module pulls no media library (no moviepy, opencv, numpy or
PIL). [`mixing.edit`](mixing.edit.html.md#module-mixing.edit) and `mixing.video.cutaways` both read their
sources through it.

```pycon
>>> from mixing.probe import probe
>>> info = probe("talk.mp4")
>>> info.fps, info.frames, info.width, info.height
(Fraction(30000, 1001), 900, 1920, 1080)
```

**Two methods, one result.** [`probe()`](#mixing.probe.probe) returns a frozen [`MediaProbe`](#mixing.probe.MediaProbe)
whose `method` says how it was read:

- `"ffprobe"` (exact): the rate is `r_frame_rate` as a
  [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction) (`30000/1001` stays exact), the frame count is
  > the counted video packets (or the container’s `nb_frames` header), the
  > video starts at its `start_time` and the sound at its first **decoded**
  > frame (an opus stream’s first packet precedes it by its pre-skip), cover
  > art (`attached_pic`) is not taken for the picture, the rotation comes from the display
  > matrix (side data) or the `rotate` tag, and `r_frame_rate` is checked
  > against `avg_frame_rate` (and the packet count against the duration) to
  > detect a variable rate.
- `"banner"` (approximate): when no `ffprobe` sits beside
  [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe) or on `PATH` (imageio-ffmpeg’s build ships
  > none), the text `ffmpeg -i` prints is parsed by [`parse_banner()`](#mixing.probe.parse_banner). The
  > rates are the printed decimals (`tbr` as `fps`, `fps` as `avg_fps`)
  > snapped by [`snap_rate()`](#mixing.probe.snap_rate), the frame count is estimated from the
  > **container** duration, both streams are assumed to start with the file, and
  > a variable rate is seen only when the two printed rates disagree.

`ffprobe=` is the seam: the default finds one ([`find_ffprobe()`](#mixing.probe.find_ffprobe)), a path
uses that binary, `None` forces the banner. A file that does not exist raises
[`MediaNotFoundError`](mixing.errors.html.md#mixing.errors.MediaNotFoundError); one ffmpeg cannot read raises
[`ProbeError`](mixing.errors.html.md#mixing.errors.ProbeError) (both are 

```
``
```

ValueError\`\`s).

### Module Attributes

| [`AUTO_FFPROBE`](#mixing.probe.AUTO_FFPROBE)   | `probe(ffprobe=AUTO_FFPROBE)` (the default) looks for an ffprobe with [`find_ffprobe()`](#mixing.probe.find_ffprobe); a path names one; `None` forces the banner.   |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`FFPROBE_METHOD`](#mixing.probe.FFPROBE_METHOD) | [`MediaProbe.method`](#mixing.probe.MediaProbe.method) of an exact (`ffprobe`) and an approximate (`ffmpeg -i` banner) reading.                                          |

### Functions

| [`probe`](#mixing.probe.probe)(media, \*[, ffprobe, count_packets, ...])   | Probe `media` with ffprobe when there is one, else with ffmpeg's banner.                                                                                                                                 |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`probe_duration`](#mixing.probe.probe_duration)(media, \*[, timeout, ffmpeg])      | The container duration of `media` in seconds, from `ffmpeg -i`, or `None`.                                                                                                                               |
| [`parse_banner`](#mixing.probe.parse_banner)(text)                                | A [`MediaProbe`](#mixing.probe.MediaProbe) from the text `ffmpeg -i FILE` prints (approximate).                                                                                       |
| [`find_ffprobe`](#mixing.probe.find_ffprobe)(\*[, ffmpeg])                        | The ffprobe beside `ffmpeg` (default [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe); a bare name is looked up on `PATH`), else on `PATH`, else `None`. |
| [`snap_rate`](#mixing.probe.snap_rate)(fps)                                    | A float frame rate as the exact rate it stands for.                                                                                                                                                      |

### Classes

| [`MediaProbe`](#mixing.probe.MediaProbe)(method[, duration, width, ...])   | What [`probe()`](#mixing.probe.probe) read about a media file.                                                                                     |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`MediaStream`](#mixing.probe.MediaStream)(kind[, codec, attached_pic])     | One stream of a media file, as [`probe()`](#mixing.probe.probe) lists them in [`MediaProbe.streams`](#mixing.probe.MediaProbe.streams). |

### mixing.probe.AUTO_FFPROBE *= 'auto'*

`probe(ffprobe=AUTO_FFPROBE)` (the default) looks for an ffprobe with
[`find_ffprobe()`](#mixing.probe.find_ffprobe); a path names one; `None` forces the banner.

### mixing.probe.FFPROBE_METHOD *= 'ffprobe'*

[`MediaProbe.method`](#mixing.probe.MediaProbe.method) of an exact (`ffprobe`) and an approximate
(`ffmpeg -i` banner) reading.

### *class* mixing.probe.MediaProbe(method, duration=None, width=None, height=None, rotation=0.0, fps=None, avg_fps=None, frames=None, packets=None, video_start=0.0, video_duration=None, audio_start=None, sample_rate=None, channels=None, variable_rate=False, has_audio=False, streams=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What [`probe()`](#mixing.probe.probe) read about a media file.

Video fields are `None` (and `has_video` false) for an audio-only file;
sound fields are `None` (and `has_audio` false) for a silent one.

#### method

`"ffprobe"` (exact) or `"banner"` (approximate, see the
module docstring).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### duration

the container’s duration (s), `None` when not stated.

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### width, height

the size of the first video stream’s decoded frames
after the display rotation (a 90/270-degree rotation swaps the
coded size, as ffmpeg’s decode does). The sample aspect ratio is
not applied: an anamorphic stream reports its storage size.

#### rotation

the display rotation in degrees, as ffmpeg reports it.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### fps

the nominal rate (`r_frame_rate`, else `avg_frame_rate`); from
the banner, its `tbr` (else its `fps`) snapped by [`snap_rate()`](#mixing.probe.snap_rate).

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction)]

#### avg_fps

`avg_frame_rate`; from the banner, its `fps`. `None`
when unset.

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction)]

#### frames

the frame count at `fps`: the video packets when the rate is
constant and they are known, else the video duration times `fps`.

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

#### packets

the video packet count, counted (`count_packets=True`) or
the container’s `nb_frames` header; `None` when unknown.

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

#### video_start

where the first video frame sits after the file’s start
(what an input seek counts from).

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### video_duration

the video stream’s own length (s).

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

#### audio_start

where the sound’s first decoded frame plays after the
file’s start (from the banner, assumed `0.0`: it states no
stream start times).

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### sample_rate, channels

of the first audio stream.

#### variable_rate

the picture is not at a constant rate (the nominal and
average rates disagree by more than `VFR_TOLERANCE`, or, from
ffprobe, the packets disagree with the duration).

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### has_audio

the file has an audio stream.

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### streams

every stream of the file, in file order ([`MediaStream`](#mixing.probe.MediaStream)).

* **Type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`MediaStream`](#mixing.probe.MediaStream), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### *property* audio_offset *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[float](https://docs.python.org/3/builtins/functions.html#float)]*

before).

* **Type:**
  How far the sound starts after the first video frame (negative

#### *property* coded_size *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)]]*

`(width, height)` as stored, before the display rotation.

#### *property* exact *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether this came from ffprobe (exact) rather than the banner.

#### *property* playback_fps *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Fraction](https://docs.python.org/3/library/fractions.html#fractions.Fraction)]*

The rate the frames actually arrive at, decoded one by one.

`fps` (`r_frame_rate`) is the lowest rate that represents every
timestamp, not the rate frames arrive at: a 24 fps film with sub-frame
holes between its shots reads `120/1` (reeleehq/an#195, cutan’s
`stream_fps`). So for a `variable_rate` stream the average rate is
the playback rate; otherwise `fps` (`avg_fps` when it is unset).

```pycon
>>> MediaProbe("ffprobe", fps=Fraction(120), avg_fps=Fraction(5760, 239),
...            variable_rate=True).playback_fps
Fraction(5760, 239)
>>> MediaProbe("ffprobe", fps=Fraction(24), avg_fps=Fraction(24)).playback_fps
Fraction(24, 1)
```

### *class* mixing.probe.MediaStream(kind, codec=None, attached_pic=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One stream of a media file, as [`probe()`](#mixing.probe.probe) lists them in [`MediaProbe.streams`](#mixing.probe.MediaProbe.streams).

#### kind

`"video"`, `"audio"`, `"subtitle"`, `"data"`, `"attachment"`
(ffprobe’s `codec_type`, the banner’s stream label, lower-cased). Cover
art is a `"video"` stream with `attached_pic`.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### codec

the codec’s short name (`"h264"`, `"aac"`), `None` when not stated.

* **Type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### attached_pic

the stream is a still picture attached to the file (an mp3’s or
m4a’s cover art), not a picture track.

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.probe.find_ffprobe(, ffmpeg=None)

The ffprobe beside `ffmpeg` (default [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe); a bare
name is looked up on `PATH`), else on `PATH`, else `None`.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### mixing.probe.parse_banner(text)

A [`MediaProbe`](#mixing.probe.MediaProbe) from the text `ffmpeg -i FILE` prints (approximate).

The banner states the container duration, the first video stream’s coded
size, its two rates (`N tbr` is ffmpeg’s guess at `r_frame_rate` and
becomes `fps`; `N fps` is `avg_frame_rate` and becomes `avg_fps`;
both snapped by [`snap_rate()`](#mixing.probe.snap_rate)), the display rotation and the first
audio stream’s rate and layout. It has no per-stream start times (both
streams are **assumed** to start with the file: `video_start` and
`audio_start` are `0.0`) and no packet count (`frames` is the
container duration times `fps`), so a variable rate is detected only
when the two printed rates disagree.

```pycon
>>> found = parse_banner('''Duration: 00:00:10.50, start: 0.0
...   Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 29.97 fps
...   Stream #0:1: Audio: aac, 48000 Hz, stereo, fltp''')
>>> found.fps, found.frames, (found.width, found.height), found.channels
(Fraction(30000, 1001), 315, (1920, 1080), 2)
```

* **Return type:**
  [`MediaProbe`](#mixing.probe.MediaProbe)

### mixing.probe.probe(media, , ffprobe='auto', count_packets=True, ffmpeg=None, timeout=None)

Probe `media` with ffprobe when there is one, else with ffmpeg’s banner.

* **Parameters:**
  * **media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – the file (or any location ffmpeg opens).
  * **ffprobe** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the ffprobe binary. [`AUTO_FFPROBE`](#mixing.probe.AUTO_FFPROBE) (default) finds one
    with [`find_ffprobe()`](#mixing.probe.find_ffprobe) (beside `ffmpeg`); `None` forces the
    approximate banner ([`parse_banner()`](#mixing.probe.parse_banner)).
  * **count_packets** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – have ffprobe count the video packets (reads, but does
    not decode, the whole file) for an exact frame count. `False`
    uses the container’s `nb_frames` header, which mkv lacks.
  * **ffmpeg** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the ffmpeg whose banner is read, and beside which ffprobe is
    looked for (default [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe); `"ffmpeg"` for
    the one on `PATH`), as `mixing.decode` takes it.
  * **timeout** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – seconds the whole probe may take (every ffprobe/ffmpeg run it
    makes); the run still going then is killed, with what it started,
    and [`FfmpegTimeout`](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegTimeout) is raised. `None`
    (default) for no bound.
* **Raises:**
  * [**MediaNotFoundError**](mixing.errors.html.md#mixing.errors.MediaNotFoundError) – `media` does not exist (a `FileNotFoundError`
        too, so tell it from a missing `ffmpeg` by this class).
  * [**ProbeError**](mixing.errors.html.md#mixing.errors.ProbeError) – ffmpeg/ffprobe cannot read it, or the named `ffprobe`
        cannot be run.
  * [**OSError**](https://docs.python.org/3/builtins/exceptions.html#OSError) – the `ffmpeg` read for the banner cannot be run (a missing
        binary is a `FileNotFoundError`), as with `mixing.decode`.
  * [**FfmpegTimeout**](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegTimeout) – the probe ran past `timeout` (a `TimeoutError`).
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `ffprobe=""`, or `timeout` is not `None` or seconds > 0.
* **Return type:**
  [`MediaProbe`](#mixing.probe.MediaProbe)

### mixing.probe.probe_duration(media, , timeout=30.0, ffmpeg=None)

The container duration of `media` in seconds, from `ffmpeg -i`, or `None`.

The cheap question alone: [`probe()`](#mixing.probe.probe)’s banner reading (one `ffmpeg -i`,
no ffprobe lookup), and `None` rather than an error whenever the duration
cannot be had: the file is missing or unreadable, its container states no
duration, or ffmpeg is still running `timeout` seconds after it started.
Unlike [`probe()`](#mixing.probe.probe) and `mixing.decode`, whose `timeout` defaults to
no bound and raises [`FfmpegTimeout`](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegTimeout), this one is
bounded by default and answers `None`: it is the question a caller asks
before deciding anything, and must not hang it. On expiry ffmpeg is killed
with whatever it started (see [`mixing.ffmpeg.run_ffmpeg()`](mixing.ffmpeg.html.md#mixing.ffmpeg.run_ffmpeg)).

* **Parameters:**
  * **media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – the file (or any location ffmpeg opens).
  * **timeout** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – seconds to wait for ffmpeg (default
    `DEFAULT_PROBE_DURATION_TIMEOUT`); `None` (or `inf`)
    waits forever.
  * **ffmpeg** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the executable, as [`probe()`](#mixing.probe.probe) takes it.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `timeout` is not `None` or a number of seconds > 0
        (see [`mixing.ffmpeg.check_timeout()`](mixing.ffmpeg.html.md#mixing.ffmpeg.check_timeout)).
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – no ffmpeg executable can be found (see
        [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe)).
  * [**OSError**](https://docs.python.org/3/builtins/exceptions.html#OSError) – the `ffmpeg` given cannot be run.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### mixing.probe.snap_rate(fps)

A float frame rate as the exact rate it stands for.

A rate near an integer or an NTSC rate (`k * 1000 / 1001`) is taken to
mean it; anything else is approximated with a denominator of at most
`MAX_RATE_DENOMINATOR`.

```pycon
>>> snap_rate(29.97), snap_rate(25.0), snap_rate(23.976), snap_rate(12.5)
(Fraction(30000, 1001), Fraction(25, 1), Fraction(24000, 1001), Fraction(25, 2))
```

* **Return type:**
  [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction)
