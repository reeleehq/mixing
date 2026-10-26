# mixing.transcript.pipeline

High-level filler-removal pipeline.

Orchestrates: extract audio -> Scribe transcribe -> filler detection ->
ffmpeg cut -> write transcript outputs.

### Functions

| [`remove_fillers`](#mixing.transcript.pipeline.remove_fillers)(input_media, output_dir, \*[, ...])   | End-to-end filler removal.   |
|-------------------------------------------------------------------------------------------------------|------------------------------|

### Classes

| [`FillerRemovalResult`](#mixing.transcript.pipeline.FillerRemovalResult)(cleaned_media, ...[, ...])   | Paths and computed ranges produced by [`remove_fillers()`](#mixing.transcript.pipeline.remove_fillers).   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|

### *class* mixing.transcript.pipeline.FillerRemovalResult(cleaned_media, transcript_md, transcript_srt, cleaned_md, cleaned_srt, scribe_json, cuts_json, keeps_json, duration, cuts=list(), keeps=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Paths and computed ranges produced by [`remove_fillers()`](#mixing.transcript.pipeline.remove_fillers).

### mixing.transcript.pipeline.remove_fillers(input_media, output_dir, , output_media=None, fillers=frozenset({'ah', 'eh', 'er', 'erm', 'hmm', 'mhm', 'mm', 'mmm', 'uh', 'uhh', 'uhm', 'um', 'umm'}), audio_events=frozenset({'(coughs)'}), api_key=None, scribe_kwargs=None, apply_keeps_kwargs=None, keep_intermediate=True, scribe_data=None)

End-to-end filler removal.

* **Parameters:**
  * **input_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Video or audio file to clean.
  * **output_dir** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory to write transcripts and intermediate files into.
    Created if missing.
  * **output_media** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Output cleaned media path. Defaults to
    `output_dir / f"{stem}.cleaned.mp4"`.
  * **audio_events** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Override the default filler / event sets.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs API key (else env `ELEVENLABS_API_KEY` or `ELEVEN_API_KEY`).
  * **scribe_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to `transcribe()`.
  * **apply_keeps_kwargs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`object`](https://docs.python.org/3/builtins/functions.html#object)]]) – Extra kwargs forwarded to `apply_keeps()`.
  * **keep_intermediate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `False`, delete the extracted audio file.
  * **scribe_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If provided, skip the network call and use this dict
    (must match Scribe’s response shape). Useful for tests / replays.
* **Return type:**
  [`FillerRemovalResult`](#mixing.transcript.pipeline.FillerRemovalResult)
* **Returns:**
  A [`FillerRemovalResult`](#mixing.transcript.pipeline.FillerRemovalResult) with file paths and the computed
  cut/keep ranges.
