# mixing.srt

Canonical SRT / timeline parsing and formatting (pure, dependency-free).

This is the single source of truth for SubRip (`.srt`) time handling in
`mixing`. Before consolidation the same logic was reimplemented in three
places (`dubbing.srt`, `video.video_subtitles`, `transcript.formats`)
with subtly different rounding and tolerance; those modules now re-export from
here.

The vocabulary:

- A **timestamp** is the `HH:MM:SS,mmm` string SRT uses; [`seconds_to_srt_time()`](#mixing.srt.seconds_to_srt_time)
  and [`srt_time_to_seconds()`](#mixing.srt.srt_time_to_seconds) convert between it and seconds (a `float`).
- A [`Cue`](#mixing.srt.Cue) is one subtitle block: a 1-based `index`, `start`/`end`
  in seconds, and `text` (which may contain embedded newlines).

Times are always seconds (`float`) everywhere except the timestamp string.

### Module Attributes

| [`TIME_RE`](#mixing.srt.TIME_RE)               | Matches an SRT cue time line, tolerant of `,` or `.` as the ms separator.   |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`fmt_srt_time`](#mixing.srt.fmt_srt_time)(seconds) | Back-compat aliases — historically these names lived in different modules.  |

### Functions

| [`srt_time_to_seconds`](#mixing.srt.srt_time_to_seconds)(timestamp)                  | Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.                                            |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`seconds_to_srt_time`](#mixing.srt.seconds_to_srt_time)(seconds)                    | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                             |
| [`fmt_srt_time`](#mixing.srt.fmt_srt_time)(seconds)                           | Back-compat aliases — historically these names lived in different modules.                               |
| [`to_srt_time`](#mixing.srt.to_srt_time)(seconds)                            | Format a time in seconds as an SRT timestamp `HH:MM:SS,mmm`.                                             |
| [`parse_srt`](#mixing.srt.parse_srt)(srt_text)                             | Parse SRT text into a list of [`Cue`](#mixing.srt.Cue) objects.              |
| [`dump_srt`](#mixing.srt.dump_srt)(cues)                                  | Serialize cues back to SRT text, renumbering from 1.                                                     |
| [`shift_srt_timestamps`](#mixing.srt.shift_srt_timestamps)(srt_text[, shift_seconds]) | Shift every cue time line in `srt_text` by `shift_seconds`.                                              |
| [`parse_vtt`](#mixing.srt.parse_vtt)(vtt_text, \*[, clean])                | Parse WebVTT into cues; the counterpart of [`parse_srt()`](#mixing.srt.parse_srt). |
| [`parse_vtt_words`](#mixing.srt.parse_vtt_words)(vtt_text)                       | Parse a WebVTT caption file into a word list shaped like Scribe's.                                       |

### Classes

| [`Cue`](#mixing.srt.Cue)(index, start, end, text)   | One SRT subtitle cue.   |
|---------------------------------------------------------------------------------|-------------------------|

### *class* mixing.srt.Cue(index, start, end, text)

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

### mixing.srt.TIME_RE *= re.compile('(\\\\d{1,2}):(\\\\d{2}):(\\\\d{2})[,.](\\\\d{1,3})\\\\s\*-->\\\\s\*(\\\\d{1,2}):(\\\\d{2}):(\\\\d{2})[,.](\\\\d{1,3})')*

Matches an SRT cue time line, tolerant of `,` or `.` as the ms separator.

### mixing.srt.dump_srt(cues)

Serialize cues back to SRT text, renumbering from 1.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.srt.fmt_srt_time(seconds)

Back-compat aliases — historically these names lived in different modules.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.srt.parse_srt(srt_text)

Parse SRT text into a list of [`Cue`](#mixing.srt.Cue) objects.

Tolerant of blank-line spacing variations and of either `,` or `.` as
the millisecond separator. Cues without a valid time line are skipped.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](#mixing.srt.Cue)]

### mixing.srt.parse_vtt(vtt_text, , clean=True)

Parse WebVTT into cues; the counterpart of [`parse_srt()`](#mixing.srt.parse_srt).

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Cue`](#mixing.srt.Cue)]

### mixing.srt.parse_vtt_words(vtt_text)

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

### mixing.srt.seconds_to_srt_time(seconds)

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

### mixing.srt.shift_srt_timestamps(srt_text, shift_seconds=0.0)

Shift every cue time line in `srt_text` by `shift_seconds`.

Negative values shift earlier; resulting negative times clamp to zero.

```pycon
>>> srt = '1\n00:43:12,187 --> 00:43:13,817\nHello world'
>>> '00:00:00,187 --> 00:00:01,817' in shift_srt_timestamps(srt, -2592)
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.srt.srt_time_to_seconds(timestamp)

Parse an `HH:MM:SS,mmm` (or `.mmm`) SRT timestamp to seconds.

```pycon
>>> srt_time_to_seconds('00:43:12,187')
2592.187
>>> srt_time_to_seconds('00:00:01.500')
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### mixing.srt.to_srt_time(seconds)

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
