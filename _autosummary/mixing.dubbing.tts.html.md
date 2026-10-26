# mixing.dubbing.tts

ElevenLabs text-to-speech (TTS) HTTP client.

Stdlib-only — no `requests` or `elevenlabs` SDK dependency, mirroring
[`mixing.transcript.scribe`](mixing.transcript.scribe.html.md#module-mixing.transcript.scribe). Synthesizes speech from text and lists the
voices available on the account.

The default model is `eleven_multilingual_v2` — ElevenLabs’ high-quality,
multilingual model (handles English, French, and ~30 other languages with the
same voice), which is the right default for dubbing into multiple languages.

An optional on-disk cache (enabled by default here, since the same line is
often re-synthesized across runs) skips the network call when the same text +
voice + model + settings have been synthesized before.

Quick start:

```pycon
>>> from mixing.dubbing import list_voices, synthesize_to_file
>>> voices = list_voices()
>>> synthesize_to_file("Hello world", voices[0]["voice_id"], "hello.mp3")
```

### Module Attributes

| [`DFLT_MODEL_ID`](#mixing.dubbing.tts.DFLT_MODEL_ID)       | High-quality, multilingual default model (one voice speaks many languages).                                                                           |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_OUTPUT_FORMAT`](#mixing.dubbing.tts.DFLT_OUTPUT_FORMAT)  | Highest standard MP3 quality that does not require a paid tier.                                                                                       |
| [`DFLT_VOICE_SETTINGS`](#mixing.dubbing.tts.DFLT_VOICE_SETTINGS) | Voice settings for dubbing fidelity — see [`DFLT_MODEL_ID`](#mixing.dubbing.tts.DFLT_MODEL_ID) on why this is not tuned for expressiveness. |

### Functions

| [`add_shared_voice`](#mixing.dubbing.tts.add_shared_voice)(public_owner_id, voice_id, ...)   | Add a shared-library voice to the account so it can be synthesized.             |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`default_cache_dir`](#mixing.dubbing.tts.default_cache_dir)()                                | Default on-disk cache for synthesized audio.                                    |
| [`find_voice`](#mixing.dubbing.tts.find_voice)(query, \*[, api_key, voices])           | Return the first voice whose name or labels contain `query` (case-insensitive). |
| [`list_voices`](#mixing.dubbing.tts.list_voices)(\*[, api_key, timeout])                | List the voices available on the account.                                       |
| [`search_shared_voices`](#mixing.dubbing.tts.search_shared_voices)(\*[, language, ...])          | Search ElevenLabs' public *shared voice library* (not just the account).        |
| [`synthesize_to_file`](#mixing.dubbing.tts.synthesize_to_file)(text, voice_id, path, ...)      | Synthesize `text` and write the audio to `path`.                                |
| [`text_to_speech`](#mixing.dubbing.tts.text_to_speech)(text, voice_id, \*[, api_key, ...]) | Synthesize `text` to speech with ElevenLabs and return audio bytes.             |

### mixing.dubbing.tts.DFLT_MODEL_ID *= 'eleven_multilingual_v2'*

High-quality, multilingual default model (one voice speaks many languages).

**This is a DUBBING default, and deliberately not an expressive one.** A dub
has to track a performance that already exists: the timing is fixed by the
source, the delivery belongs to the original actor, and per-take variance is
a defect rather than life. `eleven_multilingual_v2` is steady and speaks
many languages in one voice, which is what that job wants.

**If you are writing NARRATION or COMMENTARY, this is the wrong tool.** Use
`braidio`, which owns expressive delivery and defaults to `eleven_v3`
so inline `[audio tags]` fire — the single biggest lever on whether a read
sounds alive (measured: plain text 70.8 Hz of pitch range, densely tagged
124.4 Hz). v2 cannot render those tags at all, so a narration built on this
default is capped at the flat end before a word is written.

### mixing.dubbing.tts.DFLT_OUTPUT_FORMAT *= 'mp3_44100_128'*

Highest standard MP3 quality that does not require a paid tier.

### mixing.dubbing.tts.DFLT_VOICE_SETTINGS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]* *= {'similarity_boost': 0.8, 'stability': 0.5, 'style': 0.0, 'use_speaker_boost': True}*

Voice settings for dubbing fidelity — see [`DFLT_MODEL_ID`](#mixing.dubbing.tts.DFLT_MODEL_ID) on why this
is not tuned for expressiveness. `stability` ~0.5 keeps the read steady;
`similarity_boost` ~0.8 keeps timbre faithful to the chosen voice.

### mixing.dubbing.tts.add_shared_voice(public_owner_id, voice_id, name, , api_key=None, timeout=60.0)

Add a shared-library voice to the account so it can be synthesized.

Idempotent-ish: if the voice (by name) is already on the account, returns
its existing id instead of erroring on a duplicate add.

* **Parameters:**
  * **public_owner_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `public_owner_id` from [`search_shared_voices()`](#mixing.dubbing.tts.search_shared_voices).
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The shared voice’s `voice_id`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name to give the added voice on the account.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The account-local `voice_id` to pass to [`text_to_speech()`](#mixing.dubbing.tts.text_to_speech).

### mixing.dubbing.tts.default_cache_dir()

Default on-disk cache for synthesized audio.

Honors `$MIXING_TTS_CACHE_DIR`, then `$XDG_CACHE_HOME`, then
`~/.cache/`. Final segment is always `mixing/tts`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.dubbing.tts.find_voice(query, , api_key=None, voices=None)

Return the first voice whose name or labels contain `query` (case-insensitive).

Convenience for picking a voice by a human name (e.g. `"Brian"`) or a
label keyword (e.g. `"french"`) without memorizing voice ids.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### mixing.dubbing.tts.list_voices(, api_key=None, timeout=60.0)

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

### mixing.dubbing.tts.search_shared_voices(, language=None, use_cases=None, gender=None, category=None, sort=None, page_size=40, api_key=None, timeout=60.0, \*\*extra_params)

Search ElevenLabs’ public *shared voice library* (not just the account).

Useful for finding a native voice in a target language (e.g. a French
advertising voice) that is not yet on the account. Add the chosen voice
with [`add_shared_voice()`](#mixing.dubbing.tts.add_shared_voice), then synthesize with the returned id.

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
  `public_owner_id` (both needed by [`add_shared_voice()`](#mixing.dubbing.tts.add_shared_voice)), plus
  `name`, `language`, `accent`, `gender`, `use_case`,
  `description`.

### mixing.dubbing.tts.synthesize_to_file(text, voice_id, path, \*\*kwargs)

Synthesize `text` and write the audio to `path`.

Thin convenience wrapper over [`text_to_speech()`](#mixing.dubbing.tts.text_to_speech). Extra keyword
arguments are forwarded unchanged.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The path written.

### mixing.dubbing.tts.text_to_speech(text, voice_id, , api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, timeout=600.0, cache=True, refresh=False, return_cache_status=False)

Synthesize `text` to speech with ElevenLabs and return audio bytes.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text to speak.
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs voice id (see [`list_voices()`](#mixing.dubbing.tts.list_voices)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model id. Defaults to `eleven_multilingual_v2`.
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs `output_format` query value, e.g.
    `mp3_44100_128` (default), `mp3_44100_192` (needs a paid
    tier), or `pcm_44100`.
  * **voice_settings** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Override the default voice settings (stability,
    similarity_boost, style, use_speaker_boost). Merged over
    [`DFLT_VOICE_SETTINGS`](#mixing.dubbing.tts.DFLT_VOICE_SETTINGS).
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional ISO-639-1 hint (e.g. `"fr"`). Only some
    models honor it; ignored by others.
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `True` (default) → use [`default_cache_dir()`](#mixing.dubbing.tts.default_cache_dir). `False`
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
