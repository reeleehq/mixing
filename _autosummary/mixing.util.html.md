# mixing.util

General utilities for mixing video, audio, etc.

### Module Attributes

| [`PACKAGE_NAME`](#mixing.util.PACKAGE_NAME)   | The distribution whose extras the install hints name.                                                                        |
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`EXTRAS`](#mixing.util.EXTRAS)         | `{extra: import names it supplies}`, each declared by the same-named extra in `pyproject.toml` (a test holds the two equal). |

### Functions

| [`copy_to_clipboard`](#mixing.util.copy_to_clipboard)(data)                      | Copy data to system clipboard.                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`db_to_gain`](#mixing.util.db_to_gain)(db)                               | Amplitude gain of `db` decibels: `10 ** (db / 20)`.                                                                                                                                       |
| [`db_to_power`](#mixing.util.db_to_power)(db)                              | Power ratio of `db` decibels: `10 ** (db / 10)`.                                                                                                                                          |
| [`extra_for`](#mixing.util.extra_for)(module)                            | The extra that supplies `module` (or a parent package of it), or `None`.                                                                                                                  |
| [`ffmpeg_exe`](#mixing.util.ffmpeg_exe)()                                 | Resolve an ffmpeg executable, preferring the pip-bundled binary.                                                                                                                          |
| [`gain_to_db`](#mixing.util.gain_to_db)(gain)                             | Decibels of an amplitude `gain` (or peak, RMS): `20 log10(gain)`.                                                                                                                         |
| [`get_path_from_clipboard`](#mixing.util.get_path_from_clipboard)()                    | Get file path from clipboard and validate it.                                                                                                                                             |
| [`has_ffmpeg`](#mixing.util.has_ffmpeg)()                                 | Return True if the `ffmpeg` binary is available on `PATH`.                                                                                                                                |
| [`install_hint`](#mixing.util.install_hint)(extra)                          | The pip line that installs `extra`.                                                                                                                                                       |
| [`power_to_db`](#mixing.util.power_to_db)(power)                           | Decibels of a power ratio (a mean square, an energy share): `10 log10`.                                                                                                                   |
| [`require_extra`](#mixing.util.require_extra)(feature, extra, \*[, modules]) | Raise an informative `ImportError` if `extra` is not installed.                                                                                                                           |
| [`require_llm_chat`](#mixing.util.require_llm_chat)(feature, \*, alternative)   | `aix.chat`, the LLM call behind `feature`, or an `ImportError` naming the `llm` extra and the `alternative` that needs no LLM.                                                            |
| [`require_media_extra`](#mixing.util.require_media_extra)(feature, \*[, modules])  | [`require_extra()`](#mixing.util.require_extra) for `media`: the moviepy / opencv / scipy / pydub backends that the base `pip install mixing` (pure modules only) lacks. |
| [`require_ml_extra`](#mixing.util.require_ml_extra)(feature, \*[, modules])     | [`require_extra()`](#mixing.util.require_extra) for `ml` (`torch` / `transformers`).                                                                                     |
| [`require_package`](#mixing.util.require_package)(package_name, \*[, extra])   | Import a package, raising an informative error if not installed.                                                                                                                          |
| [`to_seconds`](#mixing.util.to_seconds)(value, \*, unit, rate)            | Convert time value to seconds based on unit.                                                                                                                                              |

### mixing.util.EXTRAS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]]* *= {'audio': ('numpy', 'pydub', 'scipy', 'moviepy', 'cv2', 'PIL', 'burns', 'looks', 'soundfile'), 'beats': ('librosa',), 'clipboard': ('pyclip',), 'demucs': ('demucs',), 'gen': ('google.genai',), 'llm': ('aix',), 'loudness': ('numpy', 'scipy'), 'media': ('numpy', 'pydub', 'scipy', 'moviepy', 'cv2', 'PIL', 'burns', 'looks'), 'ml': ('torch', 'transformers'), 'separation': ('torch', 'torchaudio', 'huggingface_hub', 'librosa', 'soundfile', 'einops', 'rotary_embedding_torch', 'yamlargparse', 'tqdm', 'torchinfo', 'joblib'), 'voxy': ('voxy',), 'widget': ('ipywidgets', 'soundfile', 'IPython')}*

`{extra: import names it supplies}`,
each declared by the same-named extra in `pyproject.toml` (a test holds the
two equal). A name may be dotted (`google.genai`: `google` is a shared
namespace). A module in several extras (numpy) maps to the first, so the
broadest install wins; a caller that knows better names its extra
(`require_package("scipy.signal", extra=LOUDNESS_EXTRA)`). `extra_for` /
`install_hint` are public so a caller can map a missing module
(`ImportError.name`) to its install line.

* **Type:**
  The one table of mixing’s extras

### mixing.util.PACKAGE_NAME *= 'mixing'*

The distribution whose extras the install hints name.

### mixing.util.copy_to_clipboard(data)

Copy data to system clipboard.

* **Parameters:**
  **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Bytes (for binary data like images) or string to copy
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### mixing.util.db_to_gain(db)

Amplitude gain of `db` decibels: `10 ** (db / 20)`.

```pycon
>>> db_to_gain(-6.0)
0.501...
```

### mixing.util.db_to_power(db)

Power ratio of `db` decibels: `10 ** (db / 10)`.

```pycon
>>> db_to_power(20.0)
100.0
```

### mixing.util.extra_for(module)

The extra that supplies `module` (or a parent package of it), or `None`.

```pycon
>>> extra_for("scipy.signal"), extra_for("librosa"), extra_for("google.genai.types")
('media', 'beats', 'gen')
>>> extra_for("requests"), extra_for("google.protobuf")
(None, None)
```

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

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

### mixing.util.gain_to_db(gain)

Decibels of an amplitude `gain` (or peak, RMS): `20 log10(gain)`.

```pycon
>>> float(gain_to_db(10.0))
20.0
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

### mixing.util.install_hint(extra)

The pip line that installs `extra`.

```pycon
>>> install_hint("media")
"pip install 'mixing[media]'"
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.util.power_to_db(power)

Decibels of a power ratio (a mean square, an energy share): `10 log10`.

```pycon
>>> float(power_to_db(100.0))
20.0
```

### mixing.util.require_extra(feature, extra, , modules=None)

Raise an informative `ImportError` if `extra` is not installed.

Checks `modules` (default: every module the extra supplies) without
importing them; the error names the extra’s install line, and its
`name` is the first missing module.

```pycon
>>> require_extra("demo", "ml", modules=("math",))
>>> require_extra("demo", "ml", modules=("no_such_module_xyz",))
Traceback (most recent call last):
  ...
ImportError: demo requires the 'ml' extra ...
```

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### mixing.util.require_llm_chat(feature, , alternative)

`aix.chat`, the LLM call behind `feature`, or an `ImportError` naming
the `llm` extra and the `alternative` that needs no LLM.

```pycon
>>> require_llm_chat("demo", alternative="segment_fn")
```

### mixing.util.require_media_extra(feature, , modules=('numpy', 'pydub', 'scipy', 'moviepy', 'cv2', 'PIL', 'burns', 'looks'))

[`require_extra()`](#mixing.util.require_extra) for `media`: the moviepy / opencv / scipy / pydub
backends that the base `pip install mixing` (pure modules only) lacks.

```pycon
>>> require_media_extra("demo", modules=("no_such_module_xyz",))
Traceback (most recent call last):
  ...
ImportError: demo requires the 'media' extra ...
```

### mixing.util.require_ml_extra(feature, , modules=('torch', 'transformers'))

[`require_extra()`](#mixing.util.require_extra) for `ml` (`torch` / `transformers`).

### mixing.util.require_package(package_name, , extra=None)

Import a package, raising an informative error if not installed.

The hint names `extra` when given (the extra the caller belongs to: in
`mixing.loudness`, scipy comes from `loudness`, not `media`), else the
extra `EXTRAS` maps `package_name` to, else the package’s own pip line.
The error is a `ModuleNotFoundError` whose `name` is `package_name`,
so a caller can map it to its own install hint. An installed package whose
own imports fail re-raises that error unchanged, rather than claiming the
package is missing: that absent-vs-broken rule is the fleet’s shared one,
`i2.optional.optional_import` (imported here, on first call, so that
`import mixing` does not pay for `i2`).

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
