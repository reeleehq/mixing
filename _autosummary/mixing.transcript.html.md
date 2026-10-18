# mixing.transcript

Transcript-driven media editing.

Speech-to-text via ElevenLabs Scribe, filler-word detection, and
timeline-aware cuts of audio/video. Produces editable transcripts
(plain prose, SRT, word-level JSON) and applies cuts via ffmpeg.

The HTTP client uses stdlib only (no `requests` or `elevenlabs` SDK
dependency), so the package adds no required deps. The runtime needs
ffmpeg on PATH for media operations, and an ElevenLabs API key (env
var `ELEVENLABS_API_KEY` or explicit `api_key=`) for transcription.

Quick start:

```pycon
>>> from mixing.transcript import remove_fillers
>>> result = remove_fillers("input.mov", "out/")
>>> print(result.cleaned_media)
```

Lower-level building blocks:

```pycon
>>> from mixing.transcript import (
...     transcribe, build_cuts, keeps_from_cuts,
...     words_to_srt, words_to_prose, apply_keeps,
... )
```

### Functions

| [`transcribe`](#mixing.transcript.transcribe)(audio, \*[, api_key, model_id, ...])    | Transcribe `audio` with ElevenLabs Scribe.                                      |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`default_cache_dir`](#mixing.transcript.default_cache_dir)()                                | Default on-disk cache for Scribe responses.                                     |
| [`is_filler`](#mixing.transcript.is_filler)(item, \*[, fillers, audio_events])       | Return `True` if `item` is a filler word or a removable audio event.            |
| [`build_cuts`](#mixing.transcript.build_cuts)(words, \*[, fillers, ...])              | Compute time ranges to REMOVE.                                                  |
| [`keeps_from_cuts`](#mixing.transcript.keeps_from_cuts)(cuts, duration)                    | Return ranges to KEEP given the cut ranges and total duration.                  |
| [`normalize_token`](#mixing.transcript.normalize_token)(text)                              | Lowercase and strip non-alpha so `"Uh,"` -> `"uh"`.                             |
| [`fmt_srt_time`](#mixing.transcript.fmt_srt_time)(seconds)                              | Back-compat aliases — historically these names lived in different modules.      |
| [`words_to_srt`](#mixing.transcript.words_to_srt)(words, \*[, max_chars, ...])          | Render an SRT from a Scribe word list (no filler removal, no remapping).        |
| [`words_to_srt_remapped`](#mixing.transcript.words_to_srt_remapped)(words, cuts, \*[, ...])      | SRT aligned to a post-cut timeline.                                             |
| [`words_to_prose`](#mixing.transcript.words_to_prose)(words, \*[, paragraph_pause, ...])  | Render a Scribe word list as plain prose, with paragraph breaks on long pauses. |
| [`remap_time_after_cuts`](#mixing.transcript.remap_time_after_cuts)(t, cuts)                     | Map `t` from the original timeline onto the post-cut timeline.                  |
| [`extract_audio`](#mixing.transcript.extract_audio)(input_path, output_path, \*[, ...])  | Extract a mono mp3 from `input_path` (suitable for STT).                        |
| [`apply_keeps`](#mixing.transcript.apply_keeps)(input_path, output_path, keeps, \*)    | Re-encode `input_path` keeping only the listed time ranges.                     |
| [`srt_for_media`](#mixing.transcript.srt_for_media)(media, \*[, srt_path, reuse, ...])   | Return `(srt_text, srt_path)` for `media`, transcribing if needed.              |
| [`remove_fillers`](#mixing.transcript.remove_fillers)(input_media, output_dir, \*[, ...]) | End-to-end filler removal.                                                      |

### Classes

| [`FillerRemovalResult`](#mixing.transcript.FillerRemovalResult)(cleaned_media, ...[, ...])   | Paths and computed ranges produced by [`remove_fillers()`](#mixing.transcript.remove_fillers).   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|

### *class* mixing.transcript.FillerRemovalResult(cleaned_media, transcript_md, transcript_srt, cleaned_md, cleaned_srt, scribe_json, cuts_json, keeps_json, duration, cuts=list(), keeps=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Paths and computed ranges produced by [`remove_fillers()`](#mixing.transcript.remove_fillers).

### mixing.transcript.apply_keeps(input_path, output_path, keeps, , video_codec='libx264', audio_codec='aac', crf=20, preset='medium', audio_bitrate='160k', faststart=True, overwrite=True)

Re-encode `input_path` keeping only the listed time ranges.

* **Parameters:**
  * **input_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input audio/video file.
  * **output_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file (extension picks the container).
  * **keeps** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – `[{"start": float, "end": float}, ...]` time ranges in seconds.
  * **audio_bitrate** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Re-encode
    knobs forwarded to ffmpeg.
  * **faststart** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add `-movflags +faststart` (mp4 web-streaming).
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass `-y` to ffmpeg.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  The output path.

### mixing.transcript.build_cuts(words, , fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), absorb_trailing_space=True, merge_gap=0.08)

Compute time ranges to REMOVE.

* **Parameters:**
  * **words** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Scribe `words` array.
  * **fillers** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler set.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default audio-event set.
  * **absorb_trailing_space** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Extend each cut to the end of the trailing
    `"spacing"` token, which avoids leaving a stranded pause.
  * **merge_gap** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Merge adjacent cuts when their gap is below this many seconds.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  List of `{"start": float, "end": float, "label": str}` dicts,
  sorted by start time and non-overlapping.

### mixing.transcript.default_cache_dir()

Default on-disk cache for Scribe responses.

Honors `$MIXING_TRANSCRIPT_CACHE_DIR`, then `$XDG_CACHE_HOME`,
then `~/.cache/`. Final segment is always `mixing/transcript`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.transcript.extract_audio(input_path, output_path, , sample_rate=16000, channels=1, bitrate='96k', overwrite=True)

Extract a mono mp3 from `input_path` (suitable for STT).

Returns the output path.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### mixing.transcript.fmt_srt_time(seconds)

Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.

Milliseconds are *rounded* (not truncated), with carry handled correctly,
and negative inputs clamp to zero.

```pycon
>>> seconds_to_srt_time(2592.187)
'00:43:12,187'
>>> seconds_to_srt_time(1.5)
'00:00:01,500'
>>> seconds_to_srt_time(-3)
'00:00:00,000'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.is_filler(item, , fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

Return `True` if `item` is a filler word or a removable audio event.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### mixing.transcript.keeps_from_cuts(cuts, duration)

Return ranges to KEEP given the cut ranges and total duration.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### mixing.transcript.normalize_token(text)

Lowercase and strip non-alpha so `"Uh,"` -> `"uh"`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.remap_time_after_cuts(t, cuts)

Map `t` from the original timeline onto the post-cut timeline.

If `t` falls inside a cut, snaps to the moment that cut would have
started in the post-cut timeline.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.transcript.remove_fillers(input_media, output_dir, , output_media=None, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), api_key=None, scribe_kwargs=None, apply_keeps_kwargs=None, keep_intermediate=True, scribe_data=None)

End-to-end filler removal.

* **Parameters:**
  * **input_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Video or audio file to clean.
  * **output_dir** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory to write transcripts and intermediate files into.
    Created if missing.
  * **output_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Output cleaned media path. Defaults to
    `output_dir / f"{stem}.cleaned.mp4"`.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs API key (else env `ELEVENLABS_API_KEY`).
  * **scribe_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to [`transcribe()`](#mixing.transcript.transcribe).
  * **apply_keeps_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to [`apply_keeps()`](#mixing.transcript.apply_keeps).
  * **keep_intermediate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `False`, delete the extracted audio file.
  * **scribe_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If provided, skip the network call and use this dict
    (must match Scribe’s response shape). Useful for tests / replays.
* **Return type:**
  [`FillerRemovalResult`](mixing.transcript.pipeline.html.md#mixing.transcript.pipeline.FillerRemovalResult)
* **Returns:**
  A [`FillerRemovalResult`](#mixing.transcript.FillerRemovalResult) with file paths and the computed
  cut/keep ranges.

### mixing.transcript.srt_for_media(media, , srt_path=None, reuse=True, refresh=False, cache=True, max_chars=80, \*\*transcribe_kwargs)

Return `(srt_text, srt_path)` for `media`, transcribing if needed.

* **Parameters:**
  * **media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Audio or video file (Scribe extracts audio from video).
  * **srt_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Where the SRT lives/should be written. Defaults to the media
    path with an `.srt` suffix.
  * **reuse** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` (default) and the SRT already exists, read and
    return it without re-transcribing.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Force re-transcription even if the SRT exists (overwrites it).
  * **cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass-through to [`mixing.transcript.transcribe()`](#mixing.transcript.transcribe)’s on-disk
    response cache.
  * **max_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max characters per SRT cue when generating.
  * **transcribe_kwargs** – Extra args forwarded to `transcribe` (e.g.
    `language_code`, `diarize`).
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  The SRT text and the path it lives at.

### mixing.transcript.transcribe(audio, , api_key=None, model_id='scribe_v1', timestamps_granularity='word', tag_audio_events=True, diarize=False, language_code=None, extra_fields=None, timeout=600.0, cache=False, refresh=False)

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
    [`default_cache_dir()`](#mixing.transcript.default_cache_dir). A path → use that directory.
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

### mixing.transcript.words_to_prose(words, , paragraph_pause=1.2, drop_fillers=False, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

Render a Scribe word list as plain prose, with paragraph breaks on long pauses.

* **Parameters:**
  * **words** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Scribe `words` array.
  * **paragraph_pause** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Insert a blank line when the gap between two
    consecutive non-filler words exceeds this many seconds.
  * **drop_fillers** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `True`, omit filler words and removable audio events.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.words_to_srt(words, , max_chars=80, sentence_endings='.?!')

Render an SRT from a Scribe word list (no filler removal, no remapping).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.words_to_srt_remapped(words, cuts, , max_chars=80, sentence_endings='.?!', drop_fillers=True, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}))

SRT aligned to a post-cut timeline.

Each word’s timestamp is shifted earlier by the cumulative duration
of all cuts that ended before it, so the resulting SRT drops in over
the cleaned media.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Modules

| [`fillers`](mixing.transcript.fillers.html.md#module-mixing.transcript.fillers)   | Filler-word detection over Scribe-shaped word lists.         |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`formats`](mixing.transcript.formats.html.md#module-mixing.transcript.formats)   | Transcript output formats: SRT, plain prose, time remapping. |
| [`media`](mixing.transcript.media.html.md#module-mixing.transcript.media)       | ffmpeg-backed audio extraction and timeline-cut application. |
| [`persist`](mixing.transcript.persist.html.md#module-mixing.transcript.persist)   | Reuse-or-create the SRT that lives next to a media file.     |
| [`pipeline`](mixing.transcript.pipeline.html.md#module-mixing.transcript.pipeline) | High-level filler-removal pipeline.                          |
| [`scribe`](mixing.transcript.scribe.html.md#module-mixing.transcript.scribe)     | ElevenLabs Scribe (speech-to-text) HTTP client.              |
