# mixing.transcript.align

Time authored words by a recogniser’s timed words of the same audio.

A script, a cue sheet or a line of dialogue says *what* was meant to be said; a
transcriber (ElevenLabs Scribe, Whisper, …) says *when* each word it heard was
said. [`align_words()`](#mixing.transcript.align.align_words) walks the two in order ([`difflib.SequenceMatcher`](https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher))
and gives every authored token a time:

- a token a heard word matches (same [`normalize_word()`](#mixing.transcript.align.normalize_word) form) takes that
  word’s times;
- a run heard differently (`friend` heard as `fred`) is spread over the
  misheard words’ span;
- a run not heard at all is spread over the gap to the next anchor;

spreading in proportion to token length, so every token gets a plausible slot,
and the result says how many were actually anchored (`matched` / `total`,
`coverage`) and by which heard word (`anchors`).

Heard words are split into tokens the way the reference is (`split=`,
default [`tokens_of()`](#mixing.transcript.align.tokens_of)), so a heard `"twenty-five"` or `"U.S."` matches
the script’s `twenty`, `five` / `U`, `S`: each piece takes its share of
the word’s span, and anchors the word’s own index.

Stdlib only. Heard words may be `(text, start, end)` triples or mappings
(Scribe’s `text`/`start`/`end`/`type`, or Whisper’s `word`); Scribe’s
`spacing` and `audio_event` entries are skipped.

```pycon
>>> heard = [("i", 0.0, 0.1), ("did", 0.1, 0.3), ("ask", 0.6, 0.9)]
>>> align_words(["I", "did", "NOT", "ask"], heard).words
[('I', 0.0, 0.1), ('did', 0.1, 0.3), ('NOT', 0.3, 0.6), ('ask', 0.6, 0.9)]
```

### Module Attributes

| [`WordTiming`](#mixing.transcript.align.WordTiming)       | `(text, start_s, end_s)`.                                                                                                                                         |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`HeardWord`](#mixing.transcript.align.HeardWord)        | a [`WordTiming`](#mixing.transcript.align.WordTiming), or a mapping with `text` (or `word`), `start`, `end` (and optionally `type`, as in Scribe's list). |
| [`APOSTROPHES`](#mixing.transcript.align.APOSTROPHES)      | Apostrophes kept inside a word (straight and typographic).                                                                                                        |
| [`TOKEN_PATTERN`](#mixing.transcript.align.TOKEN_PATTERN)    | letters and digits, an apostrophe inside a word kept.                                                                                                             |
| [`SPOKEN_WORD_TYPE`](#mixing.transcript.align.SPOKEN_WORD_TYPE) | The Scribe word `type` that is speech (`spacing`/`audio_event` are not).                                                                                          |
| [`TEXT_KEYS`](#mixing.transcript.align.TEXT_KEYS)        | Keys a heard-word mapping may carry its text under (Scribe, Whisper).                                                                                             |
| [`TIME_DECIMALS`](#mixing.transcript.align.TIME_DECIMALS)    | Decimals kept on spread (interpolated) times, so sums do not drift visibly.                                                                                       |

### Functions

| [`align_words`](#mixing.transcript.align.align_words)(reference, heard, \*[, bounds, ...])   | Time the `reference` tokens (what was meant to be said) by a recogniser's `heard` words of the same audio.                                                                              |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`normalize_word`](#mixing.transcript.align.normalize_word)(text)                               | The form two words are matched by: [`tokens_of()`](#mixing.transcript.align.tokens_of) the case-folded `text`, typographic apostrophes made straight, joined by single spaces. |
| [`tokens_of`](#mixing.transcript.align.tokens_of)(text)                                    | The words of `text` as written: punctuation dropped, an apostrophe (straight or typographic) inside a word kept, case kept.                                                             |

### Classes

| [`AlignedWords`](#mixing.transcript.align.AlignedWords)(words, matched, total[, anchors])   | [`align_words()`](#mixing.transcript.align.align_words)'s result.   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|

### mixing.transcript.align.APOSTROPHES *= "'’"*

Apostrophes kept inside a word (straight and typographic).

### *class* mixing.transcript.align.AlignedWords(words, matched, total, anchors=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

[`align_words()`](#mixing.transcript.align.align_words)’s result.

`words` are the reference tokens timed, in reference order with
non-decreasing starts; `matched` of the `total` (every reference token,
those that normalize to nothing included) were anchored by a heard word
(the rest interpolated); `anchors` maps a reference index to the index,
in the heard list as given, of the word that anchored it.

#### *property* coverage *: [float](https://docs.python.org/3/builtins/functions.html#float)*

The share of reference tokens anchored by a heard word, 0..1.

### mixing.transcript.align.HeardWord

a [`WordTiming`](#mixing.transcript.align.WordTiming), or a mapping with `text` (or
`word`), `start`, `end` (and optionally `type`, as in Scribe’s list).

* **Type:**
  A recogniser’s word

alias of [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)

### mixing.transcript.align.SPOKEN_WORD_TYPE *= 'word'*

The Scribe word `type` that is speech (`spacing`/`audio_event` are not).

### mixing.transcript.align.TEXT_KEYS *= ('text', 'word')*

Keys a heard-word mapping may carry its text under (Scribe, Whisper).

### mixing.transcript.align.TIME_DECIMALS *= 6*

Decimals kept on spread (interpolated) times, so sums do not drift visibly.

### mixing.transcript.align.TOKEN_PATTERN *= re.compile("[^\\\\W_]+(?:['’][^\\\\W_]+)\*")*

letters and digits, an apostrophe inside a word kept.

* **Type:**
  A word as written

### mixing.transcript.align.WordTiming

`(text, start_s, end_s)`.

* **Type:**
  One timed word

alias of [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### mixing.transcript.align.align_words(reference, heard, \*, bounds=None, normalize=<function normalize_word>, split=<function tokens_of>)

Time the `reference` tokens (what was meant to be said) by a
recogniser’s `heard` words of the same audio.

`bounds` (`(start, end)` of the audio) time the ends when no anchor
does; without it the first and last heard words do. `normalize` is the
matching key (default [`normalize_word()`](#mixing.transcript.align.normalize_word): case-folded, punctuation
dropped); pass your own for a language with elisions or a phonetic key.
`split` cuts a heard word into tokens like the reference’s (default
[`tokens_of()`](#mixing.transcript.align.tokens_of); `None` keeps each heard word whole). Heard words must
come in time order (overlaps are fine); times must be finite.

The walk is `difflib`’s, roughly quadratic: about 0.2 s for 5,000 words,
4 s for 20,000; align long recordings by section.

```pycon
>>> heard = [("i", 0.0, 0.1), ("did", 0.1, 0.3), ("ask", 0.6, 0.9)]
>>> aligned = align_words(["I", "did", "NOT", "ask"], heard)
>>> aligned.words
[('I', 0.0, 0.1), ('did', 0.1, 0.3), ('NOT', 0.3, 0.6), ('ask', 0.6, 0.9)]
>>> aligned.matched, aligned.total, aligned.anchors
(3, 4, {0: 0, 1: 1, 3: 2})
```

Scribe’s word list works as is (spacing and audio events are skipped, and
`anchors` index the list as given):

```pycon
>>> scribe = [{"text": "Hi", "start": 0.0, "end": 0.2, "type": "word"},
...           {"text": " ", "start": 0.2, "end": 0.3, "type": "spacing"},
...           {"text": "there.", "start": 0.3, "end": 0.6, "type": "word"}]
>>> align_words(["hi", "there"], scribe).anchors
{0: 0, 1: 2}
```

* **Return type:**
  [`AlignedWords`](#mixing.transcript.align.AlignedWords)

### mixing.transcript.align.normalize_word(text)

The form two words are matched by: [`tokens_of()`](#mixing.transcript.align.tokens_of) the case-folded
`text`, typographic apostrophes made straight, joined by single spaces.

```pycon
>>> normalize_word(" It’s,")
"it's"
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### mixing.transcript.align.tokens_of(text)

The words of `text` as written: punctuation dropped, an apostrophe
(straight or typographic) inside a word kept, case kept. What a line’s
authored tokens are.

```pycon
>>> tokens_of("Hello, there — it’s ME!")
['Hello', 'there', 'it’s', 'ME']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
