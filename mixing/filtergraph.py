"""Pure ffmpeg filter-string builders: no media dependency, no I/O.

The stock-ffmpeg pitch shift (``asetrate`` -> ``aresample`` -> ``atempo``) and
the ``atempo`` staging rule, spelled once for every package that runs them.
:func:`mixing.audio.shift_pitch` (``method="resample"``) runs the chain with
its measured delay removed; ``an`` keys its voice effects on the exact string
(an#611), so a change to a string built here moves every such key.

>>> resample_pitch_filter(12, 44100)
'asetrate=88200,aresample=44100,atempo=0.5000000000'
>>> resample_pitch_filter(0, 44100, tempo=3.0, flush_s=0.25)
'apad=pad_dur=0.25,atempo=2.0000000000,atempo=1.5000000000'
>>> atempo_chain(0.3)
[0.5, 0.6]
"""

from __future__ import annotations

SEMITONES_PER_OCTAVE = 12.0
#: ``atempo`` accepts a factor in this range per stage (older ffmpeg builds).
ATEMPO_RANGE = (0.5, 2.0)
#: Decimals each ``atempo`` factor is written with.
ATEMPO_DECIMALS = 10


def atempo_chain(
    factor: float, *, limits: tuple[float, float] = ATEMPO_RANGE
) -> list[float]:
    """``atempo`` stages whose product is ``factor``, each within ``limits``.

    >>> atempo_chain(1.5), atempo_chain(3.0)
    ([1.5], [2.0, 1.5])
    """
    lo, hi = limits
    stages = []
    while factor > hi or factor < lo:
        step = hi if factor > hi else lo
        stages.append(step)
        factor /= step
    return [*stages, factor]


def resample_pitch_filter(
    semitones: float, fs: int, *, tempo: float = 1.0, flush_s: float = 0.0
) -> str:
    """The ``-af`` chain shifting audio at ``fs`` by ``semitones``; ``""`` for none.

    ``asetrate`` relabels the rate (pitch and speed both move; ffmpeg takes an
    integer rate, so the ratio is rounded there), ``aresample`` returns to
    ``fs``, and ``atempo`` stages undo the speed change of the rounded rate
    exactly, times ``tempo`` (``1.5``: half again as fast, pitch kept).
    ``flush_s`` seconds of silence are appended before the ``atempo`` stages
    (``apad``) so their window never drops the tail. With ``semitones == 0``
    only the tempo stages remain.

    >>> resample_pitch_filter(0, 44100)
    ''
    >>> resample_pitch_filter(-12, 16000, tempo=1.5)
    'asetrate=8000,aresample=16000,atempo=2.0000000000,atempo=1.5000000000'
    """
    if not semitones and tempo == 1.0:
        return ""
    filters, factor = [], tempo
    if semitones:
        rate = int(round(fs * 2.0 ** (semitones / SEMITONES_PER_OCTAVE)))
        filters += [f"asetrate={rate}", f"aresample={fs}"]
        factor *= fs / rate
    if flush_s:
        filters.append(f"apad=pad_dur={flush_s:g}")
    filters += [f"atempo={t:.{ATEMPO_DECIMALS}f}" for t in atempo_chain(factor)]
    return ",".join(filters)
