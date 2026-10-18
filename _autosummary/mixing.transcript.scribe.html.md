# mixing.transcript.scribe

ElevenLabs Scribe (speech-to-text) HTTP client.

Stdlib-only — no `requests` or `elevenlabs` SDK dependency, so this
module adds nothing to the project’s required deps. Returns the raw
JSON response which includes word-level timestamps (with per-word
`confidence`) when `timestamps_granularity="word"` (the default).

Optional on-disk cache: pass `cache=True` (default location) or
`cache=<path>` to skip a re-call when the same audio + params have
been transcribed before.

### Functions

| [`default_cache_dir`](#mixing.transcript.scribe.default_cache_dir)()                             | Default on-disk cache for Scribe responses.   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------|
| [`transcribe`](#mixing.transcript.scribe.transcribe)(audio, \*[, api_key, model_id, ...]) | Transcribe `audio` with ElevenLabs Scribe.    |

### mixing.transcript.scribe.default_cache_dir()

Default on-disk cache for Scribe responses.

Honors `$MIXING_TRANSCRIPT_CACHE_DIR`, then `$XDG_CACHE_HOME`,
then `~/.cache/`. Final segment is always `mixing/transcript`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.transcript.scribe.transcribe(audio, , api_key=None, model_id='scribe_v1', timestamps_granularity='word', tag_audio_events=True, diarize=False, language_code=None, extra_fields=None, timeout=600.0, cache=False, refresh=False)

Transcribe `audio` with ElevenLabs Scribe.

* **Parameters:**
  * **audio** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Path to an audio/video file, or raw bytes.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY`.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Scribe model id (currently `scribe_v1`).
  * **timestamps_granularity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"word"` (default), `"character"`, or `"none"`.
  * **tag_audio_events** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to surface `(laughs)` / `(coughs)` etc.
  * **diarize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Speaker diarization on/off.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional BCP-47 hint to skip language detection.
  * **extra_fields** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Additional multipart fields to send (forward compatible).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `False` (default) → no cache. `True` → use
    [`default_cache_dir()`](#mixing.transcript.scribe.default_cache_dir). A path → use that directory.
    The cache key is the SHA-256 of the audio bytes plus all
    request parameters that affect the response.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` and `cache` is enabled, force a re-call
    and overwrite the cached entry. Useful for invalidating
    stale entries when Scribe upgrades its model.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  The raw JSON response from ElevenLabs (a dict). When
  `timestamps_granularity="word"` (default), the response contains
  a `"words"` list where each entry has at minimum
  `{"text", "start", "end", "type", "confidence"}`. Non-word
  events (`type != "word"`) include `(laughs)` etc. when
  `tag_audio_events=True`.
* **Raises:**
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` unset.
  * [**HTTPError**](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) – Request failed (4xx/5xx).
