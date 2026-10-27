# mixing.filtergraph

Pure ffmpeg filter-string builders: no media dependency, no I/O.

The stock-ffmpeg pitch shift (`asetrate` -> `aresample` -> `atempo`) and
the `atempo` staging rule, spelled once for every package that runs them.
`mixing.audio.shift_pitch()` (`method="resample"`) runs the chain with
its measured delay removed; `an` keys its voice effects on the exact string
(an#611), so a change to a string built here moves every such key.

```pycon
>>> resample_pitch_filter(12, 44100)
'asetrate=88200,aresample=44100,atempo=0.5000000000'
>>> resample_pitch_filter(0, 44100, tempo=3.0, flush_s=0.25)
'apad=pad_dur=0.25,atempo=2.0000000000,atempo=1.5000000000'
>>> atempo_chain(0.3)
[0.5, 0.6]
```

### Module Attributes

| [`ATEMPO_RANGE`](#mixing.filtergraph.ATEMPO_RANGE)    | `atempo` accepts a factor in this range per stage (older ffmpeg builds).   |
|------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`ATEMPO_DECIMALS`](#mixing.filtergraph.ATEMPO_DECIMALS) | Decimals each `atempo` factor is written with.                             |
| [`PAD_DECIMALS`](#mixing.filtergraph.PAD_DECIMALS)    | ffmpeg refuses an exponent there).                                         |

### Functions

| [`atempo_chain`](#mixing.filtergraph.atempo_chain)(factor, \*[, limits])              | `atempo` stages whose product is `factor`, each within `limits`.      |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`resample_pitch_filter`](#mixing.filtergraph.resample_pitch_filter)(semitones, fs, \*[, ...]) | The `-af` chain shifting audio at `fs` by `semitones`; `""` for none. |

### mixing.filtergraph.ATEMPO_DECIMALS *= 10*

Decimals each `atempo` factor is written with.

### mixing.filtergraph.ATEMPO_RANGE *= (0.5, 2.0)*

`atempo` accepts a factor in this range per stage (older ffmpeg builds).

### mixing.filtergraph.PAD_DECIMALS *= 6*

ffmpeg refuses an
exponent there).

* **Type:**
  Most decimals a `pad_dur` is written with (fixed point

### mixing.filtergraph.atempo_chain(factor, , limits=(0.5, 2.0))

`atempo` stages whose product is `factor`, each within `limits`.

```pycon
>>> atempo_chain(1.5), atempo_chain(3.0)
([1.5], [2.0, 1.5])
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### mixing.filtergraph.resample_pitch_filter(semitones, fs, , tempo=1.0, flush_s=0.0)

The `-af` chain shifting audio at `fs` by `semitones`; `""` for none.

`asetrate` relabels the rate (pitch and speed both move; ffmpeg takes an
integer rate, so the ratio is rounded there), `aresample` returns to
`fs`, and `atempo` stages undo the speed change of the rounded rate
exactly, times `tempo` (`1.5`: half again as fast, pitch kept).
`flush_s` seconds of silence are appended before the `atempo` stages
(`apad`) so their window never drops the tail. With `semitones == 0`
only the tempo stages remain.

```pycon
>>> resample_pitch_filter(0, 44100)
''
>>> resample_pitch_filter(-12, 16000, tempo=1.5)
'asetrate=8000,aresample=16000,atempo=2.0000000000,atempo=1.5000000000'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
