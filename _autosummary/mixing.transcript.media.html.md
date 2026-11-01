# mixing.transcript.media

ffmpeg-backed audio extraction and timeline-cut application.

These functions run one ffmpeg each, through [`mixing.ffmpeg.run_ffmpeg()`](mixing.ffmpeg.html.md#mixing.ffmpeg.run_ffmpeg);
they have no Python dependencies beyond the stdlib. Both take `ffmpeg=` (the
executable; default [`mixing.util.ffmpeg_exe()`](mixing.util.html.md#mixing.util.ffmpeg_exe), which finds a pip-bundled
build too: `"ffmpeg"` for the one on `PATH`) and `timeout=` (seconds the
run may take; it is killed, with what it started, and
[`FfmpegTimeout`](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegTimeout) raised). A failed run raises
[`FfmpegError`](mixing.ffmpeg.html.md#mixing.ffmpeg.FfmpegError): a `RuntimeError` carrying the tail of
ffmpeg’s message, and a `subprocess.CalledProcessError` (`returncode`,
`stderr`) as before this used the shared runner.

### Functions

| [`apply_keeps`](#mixing.transcript.media.apply_keeps)(input_path, output_path, keeps, \*)   | Re-encode `input_path` keeping only the listed time ranges.   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`extract_audio`](#mixing.transcript.media.extract_audio)(input_path, output_path, \*[, ...]) | Extract a mono mp3 from `input_path` (suitable for STT).      |

### mixing.transcript.media.apply_keeps(input_path, output_path, keeps, , video_codec='libx264', audio_codec='aac', crf=20, preset='medium', audio_bitrate='160k', faststart=True, overwrite=True, ffmpeg=None, timeout=None)

Re-encode `input_path` keeping only the listed time ranges.

* **Parameters:**
  * **input_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input audio/video file.
  * **output_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file (extension picks the container).
  * **keeps** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – `[{"start": float, "end": float}, ...]` time ranges in seconds.
  * **audio_bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Re-encode
    knobs forwarded to ffmpeg.
  * **faststart** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add `-movflags +faststart` (mp4 web-streaming).
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass `-y` to ffmpeg.
  * **ffmpeg** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – As in the module docstring.
  * **timeout** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – As in the module docstring.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The output path.

### mixing.transcript.media.extract_audio(input_path, output_path, , sample_rate=16000, channels=1, bitrate='96k', overwrite=True, ffmpeg=None, timeout=None)

Extract a mono mp3 from `input_path` (suitable for STT).

Returns the output path. `ffmpeg` and `timeout` as in the module docstring.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
