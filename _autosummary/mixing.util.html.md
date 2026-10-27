# mixing.util

General utilities for mixing video, audio, etc.

### Functions

| [`copy_to_clipboard`](#mixing.util.copy_to_clipboard)(data)                     | Copy data to system clipboard.                                            |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`ffmpeg_exe`](#mixing.util.ffmpeg_exe)()                                | Resolve an ffmpeg executable, preferring the pip-bundled binary.          |
| [`get_path_from_clipboard`](#mixing.util.get_path_from_clipboard)()                   | Get file path from clipboard and validate it.                             |
| [`has_ffmpeg`](#mixing.util.has_ffmpeg)()                                | Return True if the `ffmpeg` binary is available on `PATH`.                |
| [`require_media_extra`](#mixing.util.require_media_extra)(feature, \*[, modules]) | Raise an informative `ImportError` if the `media` extra is not installed. |
| [`require_ml_extra`](#mixing.util.require_ml_extra)(feature, \*[, modules])    | Raise an informative `ImportError` if the `ml` extra is not installed.    |
| [`require_package`](#mixing.util.require_package)(package_name)               | Import a package, raising an informative error if not installed.          |
| [`to_seconds`](#mixing.util.to_seconds)(value, \*, unit, rate)           | Convert time value to seconds based on unit.                              |

### mixing.util.copy_to_clipboard(data)

Copy data to system clipboard.

* **Parameters:**
  **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bytes (for binary data like images) or string to copy
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### mixing.util.ffmpeg_exe()

Resolve an ffmpeg executable, preferring the pip-bundled binary.

Resolution order is `imageio-ffmpeg` (which itself honors the
`IMAGEIO_FFMPEG_EXE` env var, then its bundled static binary, then the
system `PATH`) and finally a bare `PATH` lookup. This is what lets
subprocess-ffmpeg features work on a machine where only `pip install`
ran — `imageio-ffmpeg` is a moviepy dependency, so it is present
wherever mixing’s video stack is.

* **Raises:**
  [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – when no ffmpeg executable can be found, naming both
      remedies (installing `imageio-ffmpeg` or a system ffmpeg).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> isinstance(ffmpeg_exe(), str)
True
```

### mixing.util.get_path_from_clipboard()

Get file path from clipboard and validate it.

Intelligently extracts file paths from various clipboard contents including:

- Direct file paths
- File paths within error messages or tracebacks
- Quoted paths

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.util.has_ffmpeg()

Return True if the `ffmpeg` binary is available on `PATH`.

Most real video work in `mixing` (concatenation, trim/pad, dimension
normalization, audio replacement) shells out to ffmpeg via moviepy.
Without it, some operations degrade silently — producing wrong-duration
or lower-quality output instead of failing. Callers should check this at
startup and refuse to run rather than emit broken media.

```pycon
>>> isinstance(has_ffmpeg(), bool)
True
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.util.require_media_extra(feature, , modules=('numpy', 'pydub', 'scipy', 'moviepy', 'cv2', 'PIL', 'burns', 'looks'))

Raise an informative `ImportError` if the `media` extra is not installed.

The base `pip install mixing` ships only the pure modules (`srt`,
`chapters`, `errors`, `util`, `egress`, `transcript`, TTS); the
moviepy / opencv / scipy / pydub backends come with `mixing[media]`.

```pycon
>>> require_media_extra("demo", modules=("math",))
>>> require_media_extra("demo", modules=("no_such_module_xyz",))
Traceback (most recent call last):
  ...
ImportError: demo requires the 'media' extra ...
```

### mixing.util.require_ml_extra(feature, , modules=('torch', 'transformers'))

Raise an informative `ImportError` if the `ml` extra is not installed.

`torch` / `transformers` (the pretrained AudioSet tagger) come with
`mixing[ml]`; nothing imports them until a learned feature is used.

```pycon
>>> require_ml_extra("demo", modules=("math",))
>>> require_ml_extra("demo", modules=("no_such_module_xyz",))
Traceback (most recent call last):
  ...
ImportError: demo requires the 'ml' extra ...
```

### mixing.util.require_package(package_name)

Import a package, raising an informative error if not installed.

Packages supplied by the `media`, `ml` or `beats` extra get a hint naming that extra.
The error’s `name` is `package_name`, so a caller can map it to its own
install hint.

```pycon
>>> math = require_package('math')
>>> math.pi
3.141592653589793
```

### mixing.util.to_seconds(value, , unit, rate)

Convert time value to seconds based on unit.

* **Parameters:**
  * **value** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Time value to convert
  * **unit** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'frames'`, `'milliseconds'`], [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'seconds'`, `'samples'`, `'milliseconds'`]]) – Unit of the value (‘seconds’, ‘frames’, ‘milliseconds’, ‘samples’)
  * **rate** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Frame rate (fps) for ‘frames’ or sample rate (Hz) for ‘samples’
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Time in seconds

### Examples

```pycon
>>> to_seconds(10, unit="seconds", rate=24)
10.0
>>> to_seconds(240, unit="frames", rate=24)
10.0
>>> to_seconds(10000, unit="milliseconds", rate=24)
10.0
```
