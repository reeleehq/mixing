# mixing.transcript.persist

Reuse-or-create the SRT that lives next to a media file.

Encodes the “transcribe once, persist, reuse” rule: the SRT lives alongside
the media (same folder, same basename, `.srt`). If it already exists it is
returned untouched — re-transcribing costs API credits and the file may have
been hand-corrected. Otherwise it is generated via ElevenLabs Scribe (whose
raw response is itself cached on disk) and written next to the media.

### Functions

| [`srt_for_media`](#mixing.transcript.persist.srt_for_media)(media, \*[, srt_path, reuse, ...])   | Return `(srt_text, srt_path)` for `media`, transcribing if needed.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|

### mixing.transcript.persist.srt_for_media(media, , srt_path=None, reuse=True, refresh=False, cache=True, max_chars=80, \*\*transcribe_kwargs)

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
