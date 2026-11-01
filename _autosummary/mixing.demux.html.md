# mixing.demux

A muxed file’s sound as WAV: [`audio_track()`](#mixing.demux.audio_track).

For the answer a video-to-audio model returns as a video (MMAudio, ThinkSound,
HunyuanVideo-Foley): the first audio stream decoded to PCM WAV at its own rate
and channel count (no resample, no downmix; 16-bit by default, `codec=` for
float), unlike [`mixing.transcript.media.extract_audio()`](mixing.transcript.media.html.md#mixing.transcript.media.extract_audio) (16 kHz mono mp3,
for speech-to-text). One ffmpeg run through [`mixing.ffmpeg.run_ffmpeg()`](mixing.ffmpeg.html.md#mixing.ffmpeg.run_ffmpeg)
(`timeout=` kills it, and what it started). Pure stdlib, so importing it
pulls no media library.

```pycon
>>> from mixing.demux import audio_track
>>> wav = audio_track(answer_mp4_bytes)
>>> wav[:4], wav[8:12]
(b'RIFF', b'WAVE')
```

### Module Attributes

| [`DEFAULT_AUDIO_TRACK_CODEC`](#mixing.demux.DEFAULT_AUDIO_TRACK_CODEC)   | 16-bit PCM (`"pcm_f32le"` keeps a float decoder's samples, aac's say, unquantized, at twice the size).    |
|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| [`DEFAULT_AUDIO_TRACK_TIMEOUT`](#mixing.demux.DEFAULT_AUDIO_TRACK_TIMEOUT) | Seconds [`audio_track()`](#mixing.demux.audio_track) lets ffmpeg run before killing it. |

### Functions

| [`audio_track`](#mixing.demux.audio_track)(media, \*[, output, codec, ...])   | The first audio stream of `media` as PCM WAV, at its own rate and channels.   |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### mixing.demux.DEFAULT_AUDIO_TRACK_CODEC *= 'pcm_s16le'*

16-bit PCM (`"pcm_f32le"` keeps a
float decoder’s samples, aac’s say, unquantized, at twice the size).

* **Type:**
  [`audio_track()`](#mixing.demux.audio_track)’s WAV sample format

### mixing.demux.DEFAULT_AUDIO_TRACK_TIMEOUT *= 120.0*

Seconds [`audio_track()`](#mixing.demux.audio_track) lets ffmpeg run before killing it.

### mixing.demux.audio_track(media, , output=None, codec='pcm_s16le', ffmpeg=None, timeout=120.0)

The first audio stream of `media` as PCM WAV, at its own rate and channels.

No resample and no downmix; the samples are quantized to `codec` (16-bit
by default: a 24-bit or float source loses precision there, `"pcm_f32le"`
keeps it). A stream with no samples gives a header-only WAV. ffmpeg writes a file, not a pipe, so the WAV header carries the
final size (a plain RIFF WAV, so a track is limited to 4 GB).

* **Parameters:**
  * **media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`bytearray`](https://docs.python.org/3/builtins/stdtypes.html#bytearray), [`memoryview`](https://docs.python.org/3/builtins/stdtypes.html#memoryview), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – the container, as bytes or a path, or a URL ffmpeg opens.
  * **output** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – the `output` protocol ([`mixing.egress`](mixing.egress.html.md#module-mixing.egress)): `None`
    (default) returns the WAV bytes; a file path writes them there, a
    directory writes `<input stem>.wav` (`audio.wav` for bytes or a
    URL) in it, both returning the `Path`; a callable gets the bytes.
  * **codec** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the WAV sample format, an ffmpeg PCM encoder
    ([`DEFAULT_AUDIO_TRACK_CODEC`](#mixing.demux.DEFAULT_AUDIO_TRACK_CODEC)).
  * **ffmpeg** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the executable (default [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe)).
  * **timeout** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – seconds ffmpeg may run before it is killed
    ([`DEFAULT_AUDIO_TRACK_TIMEOUT`](#mixing.demux.DEFAULT_AUDIO_TRACK_TIMEOUT); `None` for no bound).
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – `timeout` is not `None` or a number of seconds > 0
        (checked before anything runs).
  * [**MediaNotFoundError**](mixing.errors.html.md#mixing.errors.MediaNotFoundError) – `media` is a path to no file.
  * [**FfmpegTimeout**](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegTimeout) – ffmpeg ran past `timeout` (a `RuntimeError`).
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – ffmpeg cannot read it, it has no audio stream, or ffmpeg
        wrote no WAV.
