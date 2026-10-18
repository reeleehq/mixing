# mixing.dubbing.srt

SRT translation for dubbing — built on the canonical [`mixing.srt`](mixing.srt.html.md#module-mixing.srt).

SRT parsing/serialization and the [`Cue`](mixing.srt.html.md#mixing.srt.Cue) model live in
[`mixing.srt`](mixing.srt.html.md#module-mixing.srt); this module re-exports them and adds the dubbing-specific
piece: translating cue *text* into another language while preserving timings.

Translation is pluggable: pass any `translate_fn` that maps a list of strings
to a list of the same length. The default uses an LLM via the `aix` package
(provider-agnostic) when it is importable.

### Module Attributes

| [`TranslateFn`](#mixing.dubbing.srt.TranslateFn)   | A translator maps (texts, target_language, source_language) -> texts.   |
|----------------------------------------------------------------|-------------------------------------------------------------------------|

### Functions

| [`parse_srt`](#mixing.dubbing.srt.parse_srt)(srt_text)                            | Parse SRT text into a list of [`Cue`](#mixing.dubbing.srt.Cue) objects.   |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [`dump_srt`](#mixing.dubbing.srt.dump_srt)(cues)                                 | Serialize cues back to SRT text, renumbering from 1.                                          |
| [`srt_time_to_seconds`](#mixing.dubbing.srt.srt_time_to_seconds)(timestamp)                 | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                                 |
| [`translate_srt`](#mixing.dubbing.srt.translate_srt)(srt, target_language, \*[, ...]) | Translate the cue text of an SRT to `target_language`, keeping timings.                       |
| [`default_translate_fn`](#mixing.dubbing.srt.default_translate_fn)(texts, target_language)   | LLM-backed translator (segment-count preserving) using `aix.chat`.                            |

### Classes

| [`Cue`](#mixing.dubbing.srt.Cue)(index, start, end, text)   | One SRT subtitle cue.   |
|---------------------------------------------------------------------------------|-------------------------|

### *class* mixing.dubbing.srt.Cue(index, start, end, text)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One SRT subtitle cue.

#### index

1-based cue number.

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### start

Start time in seconds.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### end

End time in seconds.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### text

Cue text (may contain embedded newlines).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *property* duration *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Cue duration in seconds (never negative).

### mixing.dubbing.srt.TranslateFn

A translator maps (texts, target_language, source_language) -> texts.

alias of `Callable`[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `str | None`], `list[str]`]

### mixing.dubbing.srt.default_translate_fn(texts, target_language, source_language=None, , model=None)

LLM-backed translator (segment-count preserving) using `aix.chat`.

Translates all segments in a single call, returning a JSON array so the
one-to-one mapping with the input cues is preserved. Falls back to a clear
error if `aix` is not importable — pass your own `translate_fn` then.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### mixing.dubbing.srt.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.dubbing.srt.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.dubbing.srt.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]

### mixing.dubbing.srt.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.dubbing.srt.translate_srt(srt, target_language, , source_language=None, translate_fn=None)

Translate the cue text of an SRT to `target_language`, keeping timings.

* **Parameters:**
  * **srt** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]]) – SRT text or a list of [`Cue`](mixing.srt.html.md#mixing.srt.Cue) objects.
  * **target_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Human-readable target language (e.g. `"French"`).
  * **source_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional source language hint.
  * **translate_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – A callable `(texts, target_language, source_language)
    -> list[str]` returning one translation per input text, in order.
    Defaults to [`default_translate_fn()`](#mixing.dubbing.srt.default_translate_fn) (LLM-backed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Translated SRT text with the original cue timings.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – The translator returned a different number of segments than
      it was given.
