# mixing

Tools for video and audio editing.

`mixing` is organized into focused subpackages, each with a clear dependency
footprint so you only pay for what you import:

- `mixing.audio` — sliceable `Audio`, fades, crop,
  concat, overlay, alignment, segmentation (needs `pydub`).
- `mixing.video` — sliceable `Video`, crop, loop,
  audio replace/normalize, Ken Burns, thumbnails, subtitles (needs `moviepy`
  / `opencv`).
- [`mixing.transcript`](mixing.transcript.html.md#module-mixing.transcript) — speech-to-text (ElevenLabs Scribe, stdlib HTTP),
  filler removal, transcript formats (no heavy deps).
- [`mixing.dubbing`](mixing.dubbing.html.md#module-mixing.dubbing) — text-to-speech re-voicing / translation from SRT.
- [`mixing.srt`](mixing.srt.html.md#module-mixing.srt) — canonical SRT/timeline parsing & formatting (pure).
- [`mixing.chapters`](mixing.chapters.html.md#module-mixing.chapters) — transcript → chapter markers (pure; LLM optional).

**Lazy by design.** Importing `mixing` (or a light submodule such as
`mixing.chapters` / `mixing.srt`) does **not** import `moviepy` or
`opencv`. The heavy backends load only when you first touch a name that needs
them — e.g. `mixing.Video` or `from mixing.video import replace_audio`.

The top-level namespace is a curated, lazily-resolved facade over the
subpackages; see `__all__`. For the full surface of a subsystem, import
it directly (`from mixing.audio import ...`).

### Functions

| [`detect_chapters`](#mixing.detect_chapters)(transcript, \*[, duration, ...])   | Detect chapter markers from a `transcript`.                                                              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`dump_srt`](#mixing.dump_srt)(cues)                                     | Serialize cues back to SRT text, renumbering from 1.                                                     |
| [`has_ffmpeg`](#mixing.has_ffmpeg)()                                       | Return True if the `ffmpeg` binary is available on `PATH`.                                               |
| [`parse_srt`](#mixing.parse_srt)(srt_text)                                | Parse SRT text into a list of [`Cue`](#mixing.Cue) objects.              |
| [`parse_vtt`](#mixing.parse_vtt)(vtt_text, \*[, clean])                   | Parse WebVTT into cues; the counterpart of [`parse_srt()`](#mixing.parse_srt). |
| [`parse_vtt_words`](#mixing.parse_vtt_words)(vtt_text)                          | Parse a WebVTT caption file into a word list shaped like Scribe's.                                       |
| [`remove_fillers`](#mixing.remove_fillers)(input_media, output_dir, \*[, ...]) | End-to-end filler removal.                                                                               |
| [`seconds_to_srt_time`](#mixing.seconds_to_srt_time)(seconds)                       | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                             |
| [`srt_for_media`](#mixing.srt_for_media)(media, \*[, srt_path, reuse, ...])   | Return `(srt_text, srt_path)` for `media`, transcribing if needed.                                       |
| [`srt_time_to_seconds`](#mixing.srt_time_to_seconds)(timestamp)                     | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                                            |
| [`text_to_speech`](#mixing.text_to_speech)(text, voice_id, \*[, api_key, ...]) | Synthesize `text` to speech with ElevenLabs and return audio bytes.                                      |
| [`transcribe`](#mixing.transcribe)(audio, \*[, api_key, model_id, ...])    | Transcribe `audio` with ElevenLabs Scribe.                                                               |
| [`translate_srt`](#mixing.translate_srt)(srt, target_language, \*[, ...])     | Translate the cue text of an SRT to `target_language`, keeping timings.                                  |

### Classes

| [`Chapter`](#mixing.Chapter)(start, title)                          | A chapter marker: a start time (seconds) and a short title.                                              |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`Cue`](#mixing.Cue)(index, start, end, text)                   | One SRT subtitle cue.                                                                                    |
| [`FillerRemovalResult`](#mixing.FillerRemovalResult)(cleaned_media, ...[, ...]) | Paths and computed ranges produced by [`remove_fillers()`](#mixing.remove_fillers). |

### Exceptions

| [`MixingError`](#mixing.MixingError)                               | Base for every error `mixing` raises on its own behalf.                |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`WindowTooWideForClip`](#mixing.WindowTooWideForClip)(\*, clip_index, ...) | An explicit analysis window leaves a clip no second, independent look. |

### *class* mixing.Chapter(start, title)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A chapter marker: a start time (seconds) and a short title.

### *class* mixing.Cue(index, start, end, text)

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

### *class* mixing.FillerRemovalResult(cleaned_media, transcript_md, transcript_srt, cleaned_md, cleaned_srt, scribe_json, cuts_json, keeps_json, duration, cuts=list(), keeps=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Paths and computed ranges produced by [`remove_fillers()`](#mixing.remove_fillers).

### *exception* mixing.MixingError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base for every error `mixing` raises on its own behalf.

### *exception* mixing.WindowTooWideForClip(, clip_index, clip_duration_s, window_s, hop_s, max_window_s)

Bases: [`MixingError`](mixing.errors.html.md#mixing.errors.MixingError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

An explicit analysis window leaves a clip no second, independent look.

Raised by `mixing.audio.align_clips_to_reference()` when the caller passes a
`window_s` wider than [`max_window_s`](#mixing.WindowTooWideForClip.max_window_s) — the length above which the clip holds
no second INDEPENDENT window, so the consensus vote has nothing to arbitrate and
`support` comes back `None`.

Why an error and not a clamp or a silent single window: the caller asked for a
windowed vote and would otherwise receive a single whole-clip correlation whose
only signal is the ABSENCE of a support number — indistinguishable from a clip
that is genuinely too short to support at any window (issue #43). Clamping would
be worse still: an explicit `window_s` is the caller saying what a window means
for their material, and silently measuring at a different one makes `support`
incomparable across a set for a reason nothing reports.

It subclasses [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) because it is an argument that cannot be honored,
so existing `except ValueError` handlers around alignment keep working.

#### clip_index

Position of the offending clip in the `clips` sequence, or
`None` when the caller did not identify one.

#### clip_duration_s

The clip’s duration in seconds.

#### window_s

The window that was asked for.

#### hop_s

The hop in force — the caller’s, or the one derived from `window_s`.
Reported because it is part of the grid that was asked for; it is NOT part
of the bound, which does not depend on it (see
`_max_supportable_window_s()`).

#### max_window_s

The largest window that still leaves this clip a second,
independent look. The bound is inclusive, so for any clip long enough to be
worth aligning a retry at exactly this window measures — which is what makes
it worth reporting. A degenerate clip of a sample or two has no such window
at all; the value is floored at one sample there and is a lower bound rather
than a promise.

### mixing.detect_chapters(transcript, , duration=None, min_chapters=3, max_chapters=8, min_spacing=10.0, target_count=None, segment_fn=None, model=None)

Detect chapter markers from a `transcript`.

* **Parameters:**
  * **transcript** – One of — a Scribe response `dict` (with `"words"`), a
    Scribe `words` list, SRT text, or a list of cue dicts/objects
    exposing `start`/`end`/`text`.
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Media duration in seconds. Inferred from the transcript’s
    last timestamp when omitted; used to choose a sensible chapter
    count and to bound the final marker.
  * **min_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum chapters worth showing. If fewer survive the
    constraints, an **empty list** is returned.
  * **max_chapters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Upper bound on chapter count.
  * **min_spacing** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum seconds between consecutive chapters (players such
    as YouTube require >= 10s).
  * **target_count** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Desired chapter count. When omitted, scales with
    `duration` (roughly one chapter per ~1.5 min, clamped to
    `[min_chapters, max_chapters]`).
  * **segment_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Pluggable segmenter `(segments, target_count) -> [{start,
    title}]`. Defaults to `default_segment_fn()` (LLM-backed).
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional LLM model override for the default segmenter.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Chapter`](mixing.chapters.html.md#mixing.chapters.Chapter)]
* **Returns:**
  A list of [`Chapter`](#mixing.Chapter), first at `0:00`, spaced by at least
  `min_spacing` — or `[]` when the media can’t support
  `min_chapters` (e.g. it is too short).

### mixing.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.has_ffmpeg()

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

### mixing.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]

### mixing.parse_vtt(vtt_text, , clean=True)

Parse WebVTT into cues; the counterpart of [`parse_srt()`](#mixing.parse_srt).

Tolerates a missing hour field, cue settings, and `NOTE`/`STYLE` blocks.
Cue identifiers are dropped: `Cue.index` is the cue’s 1-based position in
the file. A cue with no text is kept (`text=""`). With `clean` (default)
inline timestamp and `<c>` tags are stripped, entities unescaped and all
whitespace, line breaks included, collapsed to single spaces; `clean=False`
keeps each cue’s text as written (lines joined by newlines).

```pycon
>>> [(c.start, c.end, c.text) for c in parse_vtt(
...     "WEBVTT\n\n01:02.500 --> 01:04.000\n<c>Hi</c> there")]
[(62.5, 64.0, 'Hi there')]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]

### mixing.parse_vtt_words(vtt_text)

Parse a WebVTT caption file into a word list shaped like Scribe’s.

Each entry is `{"text", "start", "end", "type": "word"}` (seconds), the same
shape [`mixing.transcript.transcribe()`](mixing.transcript.html.md#mixing.transcript.transcribe) returns under `"words"`, so the
result plugs into the word-level helpers (`words_to_srt`,
`mixing.audio.segmentation.text_before_segments()`, …).

YouTube auto-captions carry per-word timings as inline `<hh:mm:ss.mmm>`
tags; text before a line’s first tag takes the cue start (or, on a later
line of the same cue, the latest timing seen), and the rolling duplicate lines
(no inline timings) are ignored. A file with no inline timings falls back to
one entry per cue (its whole text). A word ends where the next one starts
(capped at its cue end). Tags are stripped and HTML entities unescaped.

```pycon
>>> vtt = '''WEBVTT
...
... 00:00:12.240 --> 00:00:15.509 align:start position:0%
... Music<00:00:12.639><c> with</c><00:00:12.800><c> meaning</c>
...
... 00:00:15.509 --> 00:00:15.519 align:start position:0%
... Music with meaning
... '''
>>> [(w["text"], w["start"]) for w in parse_vtt_words(vtt)]
[('Music', 12.24), ('with', 12.639), ('meaning', 12.8)]
>>> parse_vtt_words("1\n00:01.000 --> 00:02.500\nRock &amp; roll\n\nNOTE hi\n")
[{'text': 'Rock & roll', 'start': 1.0, 'end': 2.5, 'type': 'word'}]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### mixing.remove_fillers(input_media, output_dir, , output_media=None, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), api_key=None, scribe_kwargs=None, apply_keeps_kwargs=None, keep_intermediate=True, scribe_data=None)

End-to-end filler removal.

* **Parameters:**
  * **input_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Video or audio file to clean.
  * **output_dir** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory to write transcripts and intermediate files into.
    Created if missing.
  * **output_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Output cleaned media path. Defaults to
    `output_dir / f"{stem}.cleaned.mp4"`.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs API key (else env `ELEVENLABS_API_KEY` or `ELEVEN_API_KEY`).
  * **scribe_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to [`transcribe()`](#mixing.transcribe).
  * **apply_keeps_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to `apply_keeps()`.
  * **keep_intermediate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `False`, delete the extracted audio file.
  * **scribe_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If provided, skip the network call and use this dict
    (must match Scribe’s response shape). Useful for tests / replays.
* **Return type:**
  [`FillerRemovalResult`](mixing.transcript.pipeline.html.md#mixing.transcript.pipeline.FillerRemovalResult)
* **Returns:**
  A [`FillerRemovalResult`](#mixing.FillerRemovalResult) with file paths and the computed
  cut/keep ranges.

### mixing.seconds_to_srt_time(seconds)

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

### mixing.srt_for_media(media, , srt_path=None, reuse=True, refresh=False, cache=True, max_chars=80, \*\*transcribe_kwargs)

Return `(srt_text, srt_path)` for `media`, transcribing if needed.

* **Parameters:**
  * **media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Audio or video file (Scribe extracts audio from video).
  * **srt_path** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Where the SRT lives/should be written. Defaults to the media
    path with an `.srt` suffix.
  * **reuse** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `True` (default) and the SRT already exists, read and
    return it without re-transcribing.
  * **refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Force re-transcription even if the SRT exists (overwrites it).
  * **cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass-through to [`mixing.transcript.transcribe()`](mixing.transcript.html.md#mixing.transcript.transcribe)’s on-disk
    response cache.
  * **max_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max characters per SRT cue when generating.
  * **transcribe_kwargs** – Extra args forwarded to `transcribe` (e.g.
    `language_code`, `diarize`).
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
* **Returns:**
  The SRT text and the path it lives at.

### mixing.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.text_to_speech(text, voice_id, , api_key=None, model_id='eleven_multilingual_v2', output_format='mp3_44100_128', voice_settings=None, language_code=None, timeout=600.0, cache=True, refresh=False, return_cache_status=False)

Synthesize `text` to speech with ElevenLabs and return audio bytes.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text to speak.
  * **voice_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs voice id (see `list_voices()`).
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
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `True` (default) → use `default_cache_dir()`. `False`
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

### mixing.transcribe(audio, , api_key=None, model_id='scribe_v1', timestamps_granularity='word', tag_audio_events=True, diarize=False, language_code=None, extra_fields=None, timeout=600.0, cache=False, refresh=False)

Transcribe `audio` with ElevenLabs Scribe.

* **Parameters:**
  * **audio** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Path to an audio/video file, or raw bytes.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – API key. Falls back to env var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`).
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Scribe model id (currently `scribe_v1`).
  * **timestamps_granularity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"word"` (default), `"character"`, or `"none"`.
  * **tag_audio_events** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to surface `(laughs)` / `(coughs)` etc.
  * **diarize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Speaker diarization on/off.
  * **language_code** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional BCP-47 hint to skip language detection.
  * **extra_fields** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Additional multipart fields to send (forward compatible).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Total request timeout in seconds.
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – `False` (default) → no cache. `True` → use
    `default_cache_dir()`. A path → use that directory.
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
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – No API key supplied and `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`) unset.
  * [**HTTPError**](https://docs.python.org/3/library/urllib.error.html#urllib.error.HTTPError) – Request failed (4xx/5xx).

### mixing.translate_srt(srt, target_language, , source_language=None, translate_fn=None)

Translate the cue text of an SRT to `target_language`, keeping timings.

* **Parameters:**
  * **srt** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Cue`](mixing.srt.html.md#mixing.srt.Cue)]]) – SRT text or a list of [`Cue`](mixing.srt.html.md#mixing.srt.Cue) objects.
  * **target_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Human-readable target language (e.g. `"French"`).
  * **source_language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional source language hint.
  * **translate_fn** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – A callable `(texts, target_language, source_language)
    -> list[str]` returning one translation per input text, in order.
    Defaults to `default_translate_fn()` (LLM-backed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Translated SRT text with the original cue timings.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – The translator returned a different number of segments than
      it was given.

### Modules

| [`chapters`](mixing.chapters.html.md#module-mixing.chapters)       | Detect chapter markers from a transcript — platform-agnostic.                      |
|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`dubbing`](mixing.dubbing.html.md#module-mixing.dubbing)         | Text-to-speech dubbing: re-voice or translate a video from its SRT.                |
| [`egress`](mixing.egress.html.md#module-mixing.egress)           | Canonical *egress*: route a produced result to its destination.                    |
| [`errors`](mixing.errors.html.md#module-mixing.errors)           | Typed errors `mixing` raises — the home for exceptions a caller may want to catch. |
| [`srt`](mixing.srt.html.md#module-mixing.srt)                 | Canonical SRT / timeline parsing and formatting (pure, dependency-free).           |
| [`transcript`](mixing.transcript.html.md#module-mixing.transcript)   | Transcript-driven media editing.                                                   |
| [`util`](mixing.util.html.md#module-mixing.util)               | General utilities for mixing video, audio, etc.                                    |
| [`filtergraph`](mixing.filtergraph.html.md#module-mixing.filtergraph) | Pure ffmpeg filter-string builders: no media dependency, no I/O.                   |
