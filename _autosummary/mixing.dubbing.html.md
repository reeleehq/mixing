# mixing.dubbing

Text-to-speech dubbing: re-voice or translate a video from its SRT.

The transcript side of [`mixing`](mixing.html.md#module-mixing) turns speech into SRT (ElevenLabs
Scribe); this subpackage closes the loop — turning SRT back into speech with
an ElevenLabs voice and muxing it over the video, optionally translating the
SRT first to produce a foreign-language dub.

Building blocks:

> - [`text_to_speech()`](#mixing.dubbing.text_to_speech) / [`synthesize_to_file()`](#mixing.dubbing.synthesize_to_file) — ElevenLabs TTS.
> - [`list_voices()`](#mixing.dubbing.list_voices) / [`find_voice()`](#mixing.dubbing.find_voice) — discover account voices.
> - [`parse_srt()`](#mixing.dubbing.parse_srt) / [`dump_srt()`](#mixing.dubbing.dump_srt) / [`translate_srt()`](#mixing.dubbing.translate_srt) — SRT I/O
>   and LLM-backed translation.
> - `dub_video_from_srt()` — the end-to-end pipeline.

The HTTP clients use stdlib only (no `requests` / `elevenlabs` SDK), so
this adds no required deps. Needs ffmpeg on PATH and an ElevenLabs API key
(env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`) or explicit `api_key=`). Translation needs
either the `aix` package (`pip install 'mixing[llm]'`) or a custom
`translate_fn`.

**Lazy by design.** Like the top-level [`mixing`](mixing.html.md#module-mixing) facade, this subpackage
uses PEP 562 `__getattr__` so that the TTS / SRT building blocks
([`text_to_speech()`](#mixing.dubbing.text_to_speech), [`list_voices()`](#mixing.dubbing.list_voices), [`parse_srt()`](#mixing.dubbing.parse_srt),
[`Cue`](#mixing.dubbing.Cue), …) import with **no** `moviepy` dependency. Only
`dub_video_from_srt()` — the end-to-end pipeline that muxes audio over a
video — pulls `moviepy`, and only when it is first accessed.

Quick start:

```pycon
>>> from mixing.dubbing import dub_video_from_srt, translate_srt
>>> dub_video_from_srt("promo.mp4", "promo.srt", voice_id="...", output="promo.en.mp4")
>>> fr_srt = translate_srt(open("promo.srt").read(), "French")
```

### Functions

| [`default_cache_dir`](#mixing.dubbing.default_cache_dir)()                                | Default on-disk cache for synthesized audio.                                                |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`text_to_speech`](#mixing.dubbing.text_to_speech)(text, voice_id, \*[, api_key, ...]) | Synthesize `text` to speech with ElevenLabs and return audio bytes.                         |
| [`synthesize_to_file`](#mixing.dubbing.synthesize_to_file)(text, voice_id, path, ...)      | Synthesize `text` and write the audio to `path`.                                            |
| [`list_voices`](#mixing.dubbing.list_voices)(\*[, api_key, timeout])                | List the voices available on the account.                                                   |
| [`find_voice`](#mixing.dubbing.find_voice)(query, \*[, api_key, voices])           | Return the first voice whose name or labels contain `query` (case-insensitive).             |
| [`search_shared_voices`](#mixing.dubbing.search_shared_voices)(\*[, language, ...])          | Search ElevenLabs' public *shared voice library* (not just the account).                    |
| [`add_shared_voice`](#mixing.dubbing.add_shared_voice)(public_owner_id, voice_id, ...)   | Add a shared-library voice to the account so it can be synthesized.                         |
| [`parse_srt`](#mixing.dubbing.parse_srt)(srt_text)                                | Parse SRT text into a list of [`Cue`](#mixing.dubbing.Cue) objects. |
| [`dump_srt`](#mixing.dubbing.dump_srt)(cues)                                     | Serialize cues back to SRT text, renumbering from 1.                                        |
| [`srt_time_to_seconds`](#mixing.dubbing.srt_time_to_seconds)(timestamp)                     | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                               |
| [`translate_srt`](#mixing.dubbing.translate_srt)(srt, target_language, \*[, ...])     | Translate the cue text of an SRT to `target_language`, keeping timings.                     |
| [`default_translate_fn`](#mixing.dubbing.default_translate_fn)(texts, target_language)       | LLM-backed translator (segment-count preserving) using `aix.chat`.                          |

### Classes

| [`Cue`](#mixing.dubbing.Cue)(index, start, end, text)   | One SRT subtitle cue.   |
|---------------------------------------------------------------------------------|-------------------------|

### *class* mixing.dubbing.Cue(index, start, end, text)

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

### mixing.dubbing.add_shared_voice(public_owner_id, voice_id, name, , api_key=None, timeout=60.0)

Add a shared-library voice to the account so it can be synthesized.

Idempotent-ish: if the voice (by name) is already on the account, returns
its existing id instead of erroring on a duplicate add.

* **Parameters:**
  * **public_owner_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `public_owner_id` from [`search_shared_voices()`](#mixing.dubbing.search_shared_voices).
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The shared voice’s `voice_id`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name to give the added voice on the account.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The account-local `voice_id` to pass to [`text_to_speech()`](#mixing.dubbing.text_to_speech).

### mixing.dubbing.default_cache_dir()

Default on-disk cache for synthesized audio.

Honors `$MIXING_TTS_CACHE_DIR`, then `$XDG_CACHE_HOME`, then
`~/.cache/`. Final segment is always `mixing/tts`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.dubbing.default_translate_fn(texts, target_language, source_language=None, , model=None)

LLM-backed translator (segment-count preserving) using `aix.chat`.

Translates all segments in a single call, returning a JSON array so the
one-to-one mapping with the input cues is preserved. Falls back to a clear
error if `aix` is not importable — pass your own `translate_fn` then.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### mixing.dubbing.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.dubbing.find_voice(query, , api_key=None, voices=None)

Return the first voice whose name or labels contain `query` (case-insensitive).

Convenience for picking a voice by a human name (e.g. `"Brian"`) or a
label keyword (e.g. `"french"`) without memorizing voice ids.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### mixing.dubbing.list_voices(, api_key=None, timeout=60.0)

List the voices available on the account.

* **Parameters:**
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Request timeout in seconds.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  A list of voice dicts. Each has at least `voice_id`, `name`,
  `category`, and a `labels` dict (`accent`, `description`,
  `age`, `gender`, `use_case`) describing the voice.
* **Raises:**
  [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`) unset.

### mixing.dubbing.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.dubbing.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]

### mixing.dubbing.search_shared_voices(, language=None, use_cases=None, gender=None, category=None, sort=None, page_size=40, api_key=None, timeout=60.0, \*\*extra_params)

Search ElevenLabs’ public *shared voice library* (not just the account).

Useful for finding a native voice in a target language (e.g. a French
advertising voice) that is not yet on the account. Add the chosen voice
with [`add_shared_voice()`](#mixing.dubbing.add_shared_voice), then synthesize with the returned id.

* **Parameters:**
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ISO-639-1 language filter, e.g. `"fr"`.
  * **use_cases** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – e.g. `"advertisement"`, `"narrative_story"`.
  * **gender** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `"male"` / `"female"` / `"neutral"`.
  * **category** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – e.g. `"professional"`, `"high_quality"`.
  * **sort** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – e.g. `"trending"`.
  * **page_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max results.
  * **extra_params** – Any other query params the endpoint accepts.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  A list of shared-voice dicts. Each has `voice_id` and
  `public_owner_id` (both needed by [`add_shared_voice()`](#mixing.dubbing.add_shared_voice)), plus
  `name`, `language`, `accent`, `gender`, `use_case`,
  `description`.

### mixing.dubbing.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.dubbing.synthesize_to_file(text, voice_id, path, \*\*kwargs)

Synthesize `text` and write the audio to `path`.

Thin convenience wrapper over [`text_to_speech()`](#mixing.dubbing.text_to_speech). Extra keyword
arguments are forwarded unchanged.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The path written.

### mixing.dubbing.text_to_speech(text, voice_id, , api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, timeout=600.0, cache=True, refresh=False, return_cache_status=False)

Synthesize `text` to speech with ElevenLabs and return audio bytes.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text to speak.
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs voice id (see [`list_voices()`](#mixing.dubbing.list_voices)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model id. Defaults to `eleven_multilingual_v2`.
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs `output_format` query value, e.g.
    `mp3_44100_128` (default), `mp3_44100_192` (needs a paid
    tier), or `pcm_44100`.
  * **voice_settings** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Override the default voice settings (stability,
    similarity_boost, style, use_speaker_boost). Merged over
    `DFLT_VOICE_SETTINGS`.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional ISO-639-1 hint (e.g. `"fr"`). Only some
    models honor it; ignored by others.
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `True` (default) → use [`default_cache_dir()`](#mixing.dubbing.default_cache_dir). `False`
    → no cache. A path → use that directory. The cache key is a
    SHA-256 of the text plus every parameter that affects the audio.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` and `cache` is enabled, force a re-call and
    overwrite the cached entry.
  * **return_cache_status** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True`, return `(audio, was_cached)` instead
    of just `audio`. `was_cached` is `True` iff the bytes came from the
    on-disk cache (no ElevenLabs call = no spend), so a caller can attribute
    real cost. Default `False` keeps the `bytes` return.
* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]
* **Returns:**
  Raw audio bytes in `output_format` (MP3 by default) — or, when
  `return_cache_status` is `True`, a `(audio, was_cached)` tuple.
* **Raises:**
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`) unset.
  * [**HTTPError**](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) – Request failed (4xx/5xx).

### mixing.dubbing.translate_srt(srt, target_language, , source_language=None, translate_fn=None)

Translate the cue text of an SRT to `target_language`, keeping timings.

* **Parameters:**
  * **srt** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]]) – SRT text or a list of [`Cue`](mixing.srt.html.md#mixing.srt.Cue) objects.
  * **target_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Human-readable target language (e.g. `"French"`).
  * **source_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional source language hint.
  * **translate_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – A callable `(texts, target_language, source_language)
    -> list[str]` returning one translation per input text, in order.
    Defaults to [`default_translate_fn()`](#mixing.dubbing.default_translate_fn) (LLM-backed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Translated SRT text with the original cue timings.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – The translator returned a different number of segments than
      it was given.

### Modules

| [`srt`](mixing.dubbing.srt.html.md#module-mixing.dubbing.srt)   | SRT translation for dubbing — built on the canonical [`mixing.srt`](mixing.srt.html.md#module-mixing.srt).   |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [`tts`](mixing.dubbing.tts.html.md#module-mixing.dubbing.tts)   | ElevenLabs text-to-speech (TTS) HTTP client.                                                                                          |
