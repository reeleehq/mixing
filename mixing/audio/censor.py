"""Bleep words: mute the voice over each word, lay a tone there, keep the rest.

A radio edit of a song, a clean cut of an interview: chosen words beeped, everything
else untouched. :func:`bleep` takes the words' spans (from a forced aligner; Whisper's
word times are off by up to ~0.3 s on sung vocals, MMS forced alignment of the phrase
is what gave usable bounds) and

1. **widens** each span to where the voice's energy falls ``widen_floor_db`` below
   the word's peak (:func:`widen_spans`), so a held sung word's tail is covered too,
   bounded by ``widen_max_s`` and by the word's ``max_end`` (the next word's onset:
   without that bound, the widening swallows the next word in connected speech);
2. **mutes** the voice there, with linear ramps of ``fade_s`` outside each span:
   :func:`~mixing.audio.separation.remove_talk` with :data:`MUTE` as its separator
   (everything goes to "speech"), or with a real separator to keep the music of a
   single mix under the beep;
3. lays a **tone** (:func:`beep_track`) over each widened span at the RMS of the word
   itself (not of the widened span, whose quiet tail would make a quiet beep) as
   the mute removed it (so with a separator, the speech, not the music under it),
   ``beep_db`` relative to it;
4. adds ``backing`` (the other stems), untouched, when the voice was given as a stem.

So the three ways to use it are: a vocal stem plus ``backing=`` (the stems a separator
already made), a single mix with ``separator=`` (split only inside the spans), or a
single mix alone (everything in the span is replaced by the tone).

>>> import numpy as np
>>> voice = np.zeros(8000); voice[2400:4000] = 0.5
>>> out = bleep(voice, [(0.3, 0.5)], sr=8000, widen_floor_db=None, fade_s=0.0)
>>> bool(np.abs(out[2400:4000]).max() > 0.5), float(np.abs(out[:2400]).max())
(True, 0.0)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from .samples import AudioLike, float_input
from .separation import (
    SeparationStrategy,
    SpeechSeparator,
    remove_talk,
)
from .spans import Span, as_spans, is_number_pair

#: Frequency of the beep (Hz), the broadcast censor tone.
BEEP_TONE_HZ = 1000.0
#: Linear ramp (s) of the mute outside each span and of the tone inside it: no click.
BLEEP_FADE_S = 0.01
#: A span is widened while the voice's energy stays above this, in dB under the
#: word's peak (#169: -24 dB covered a held sung word's tail).
WIDEN_FLOOR_DB = -24.0
#: How far (s) a span may be widened before its start and after its end.
WIDEN_MAX_S = (0.06, 0.30)
#: Hop (s) of the RMS envelope the widening walks on.
WIDEN_HOP_S = 0.005
#: Suffix of a written result's name (``<stem>_bleeped.wav``).
OUTPUT_SUFFIX = "_bleeped.wav"


def _mute(samples: np.ndarray, sr: int):
    return samples, np.zeros_like(samples)


#: The trivial separator: everything is "speech", so the span is silenced.
MUTE = SeparationStrategy(
    name="mute", separate=_mute, license="n/a", commercial_ok=True, context_s=0.0
)


@dataclass(frozen=True)
class Word:
    """A span to bleep: ``start``/``end`` in seconds, and an optional ``max_end``
    the widening never passes (the next word's onset)."""

    start: float
    end: float
    max_end: Optional[float] = None


def as_words(spans) -> list[Word]:
    """``spans`` as :class:`Word`\\ s: anything :func:`~mixing.audio.spans.as_spans`
    takes; a mapping (an aligner's word) or an object with a ``max_end`` keeps it.

    >>> as_words([(1, 2), {"word": "x", "start": 3, "end": 4, "max_end": 4.2}])
    [Word(start=1.0, end=2.0, max_end=None), Word(start=3.0, end=4.0, max_end=4.2)]
    """
    if isinstance(spans, Word):
        return [spans]
    is_list = isinstance(spans, (list, tuple)) and not is_number_pair(spans)
    items = spans if is_list else [spans]
    words = []
    for item in items:
        if isinstance(item, Word):
            words.append(item)
            continue
        bound = (
            item.get("max_end")
            if isinstance(item, Mapping)
            else getattr(item, "max_end", None)
        )
        for start, end in as_spans(item):
            words.append(Word(start, end, None if bound is None else float(bound)))
    return words


def _envelope(x: np.ndarray, hop: int) -> np.ndarray:
    """RMS over the channels' mean power, per ``hop`` frames (the tail is dropped)."""
    x = np.asarray(x).reshape(len(x), -1)  # mono (n,) is one channel
    k = len(x) // hop
    power = np.square(x[: k * hop]).mean(axis=1)
    return np.sqrt(power.reshape(k, hop).mean(axis=1))


def widen_spans(
    samples: np.ndarray,
    sr: int,
    words: Sequence[Word],
    *,
    floor_db: float = WIDEN_FLOOR_DB,
    max_s: tuple[float, float] = WIDEN_MAX_S,
    hop_s: float = WIDEN_HOP_S,
) -> list[Span]:
    """Each word's span, widened while ``samples`` (the voice, ``(n,)`` or
    ``(n, channels)``) stays above ``floor_db`` under the word's peak, at most
    ``max_s`` (before, after) and never past the word's ``max_end`` nor outside
    the signal. Give it the voice alone: on a whole mix the music rarely drops
    ``floor_db`` below a word, so every span widens by the whole ``max_s``.

    Times are on the envelope's ``hop_s`` grid where widened; a word that cannot
    be widened keeps its own bounds.
    """
    hop = max(1, int(round(hop_s * sr)))
    env = _envelope(samples, hop)
    step = hop / sr
    out = []
    for w in words:
        i, j = int(w.start / step), int(np.ceil(w.end / step))
        inside = env[max(0, i) : min(len(env), j)]
        if not inside.size or inside.max() <= 0:
            out.append((w.start, w.end))
            continue
        floor = inside.max() * 10.0 ** (floor_db / 20.0)
        lo = max(0, i - int(round(max_s[0] / step)))
        hi = min(len(env), j + int(round(max_s[1] / step)))
        if w.max_end is not None:
            hi = min(hi, max(j, int(w.max_end / step)))
        while i > lo and env[i - 1] > floor:
            i -= 1
        while j < hi and env[j] > floor:
            j += 1
        start = min(w.start, i * step)
        end = max(w.end, j * step)
        if w.max_end is not None:
            end = max(w.end, min(end, w.max_end))
        out.append((max(0.0, start), min(len(samples) / sr, end)))
    return out


def _merge(beeps: Sequence[tuple[Span, float]]):
    """Overlapping spans merged (the louder level kept), so tones never beat."""
    merged: list[list] = []
    for (s, e), level in sorted(beeps):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], level)
        else:
            merged.append([s, e, level])
    return [((s, e), level) for s, e, level in merged]


def beep_track(
    n: int,
    sr: int,
    beeps: Sequence[tuple[Span, float]],
    *,
    tone_hz: float = BEEP_TONE_HZ,
    fade_s: float = BLEEP_FADE_S,
) -> np.ndarray:
    """``n`` mono frames: for each ``((start, end), rms)`` in ``beeps``, a sine at
    that RMS over the span (clipped to the ``n`` frames), faded over ``fade_s``
    inside it; silence elsewhere. Overlapping spans merge (the louder level)."""
    track = np.zeros(n)
    for (s, e), level in _merge(beeps):
        a, b = _frames((s, e), sr, n)
        if b <= a:
            continue
        tone = (
            np.sqrt(2.0) * level * np.sin(2 * np.pi * tone_hz * np.arange(b - a) / sr)
        )
        f = min(int(round(fade_s * sr)), (b - a) // 2)
        if f:
            ramp = np.linspace(0.0, 1.0, f, endpoint=False)
            tone[:f] *= ramp
            tone[-f:] *= ramp[::-1]
        track[a:b] = tone
    return track


def _frames(span: Span, sr: int, n: int) -> tuple[int, int]:
    """``span`` (s) as a frame range clipped to ``[0, n]`` (a span may overhang)."""
    a = min(n, max(0, int(round(span[0] * sr))))
    return a, min(n, max(a, int(round(span[1] * sr))))


def _rms(x: np.ndarray, frames: tuple[int, int]) -> float:
    seg = x[frames[0] : frames[1]]
    return float(np.sqrt(np.square(seg).mean())) if seg.size else 0.0


def _fit_backing(backing: AudioLike, sr: int, n: int, channels: int):
    # an array backing is at the voice's rate; a file's own rate must match
    back = float_input(backing, sr if isinstance(backing, np.ndarray) else None)
    if back.sr != sr:
        raise ValueError(f"backing is at {back.sr} Hz, the voice at {sr} Hz")
    y = back.samples
    if y.shape[1] not in (1, channels):
        raise ValueError(f"backing has {y.shape[1]} channels, the voice {channels}")
    if len(y) != n:
        raise ValueError(
            f"backing is {len(y)} frames, the voice {n}: give stems of one length"
        )
    return y


def bleep(
    audio: AudioLike,
    spans,
    *,
    backing: Optional[AudioLike] = None,
    separator: Union[None, str, SeparationStrategy, SpeechSeparator] = None,
    sr: Optional[int] = None,
    tone_hz: float = BEEP_TONE_HZ,
    beep_db: float = 0.0,
    fade_s: float = BLEEP_FADE_S,
    widen_floor_db: Optional[float] = WIDEN_FLOOR_DB,
    widen_max_s: tuple[float, float] = WIDEN_MAX_S,
    output: Output = None,
    **save_kwargs: Any,
):
    """``audio`` with the words in ``spans`` beeped: the voice muted, a tone laid there.

    Args:
        audio: the voice: a vocal stem, or a whole mix (an :class:`Audio`, a path,
            an ``AudioSamples`` view, or samples with ``sr``).
        spans: the words, in seconds: anything :func:`as_words` takes (pairs,
            ``Segment``\\ s, an aligner's ``{"start", "end", "max_end"}`` words).
        backing: the other stems (same rate and length, mono or the voice's
            channels), added back untouched; the result is the full mix.
        separator: with a whole mix and no ``backing``, splits the mix inside the
            spans only and keeps its music under the beep (a
            :data:`~mixing.audio.separation.SEPARATORS` key, a strategy or a
            callable). ``None``: the span is silenced (:data:`MUTE`).
        sr: the sample rate of array inputs.
        tone_hz: the tone's frequency.
        beep_db: the tone's RMS relative to the word's own RMS: of the voice
            the mute removed there (the vocal stem; with a ``separator``, its
            speech estimate; a whole mix alone, the mix). A silent word gets a
            silent beep.
        fade_s: linear ramp of the mute (outside each span) and of the tone
            (inside it).
        widen_floor_db: see :func:`widen_spans`; ``None`` keeps the spans as
            given. Widening measures ``audio``, so it wants a vocal stem: on a
            whole mix (with or without ``separator``) pass ``None`` and spans
            that already cover the words.
        widen_max_s: the widening's limit (before, after), in seconds.
        output: ``None`` (return the result in the input's kind), a path, a
            directory (``<stem>_bleeped.wav``) or a callable; see
            :mod:`mixing.egress`.
        **save_kwargs: passed to ``Audio.save`` when ``output`` is a path.

    Returns:
        The input's kind (an ``Audio``, or an array of its shape and dtype; values
        beyond full scale saturate in integer kinds), or what ``output`` makes of it.
    """
    if fade_s < 0:
        raise ValueError(f"fade_s must be >= 0, got {fade_s}")
    if backing is not None and separator is not None:
        raise ValueError(
            "give backing= (the stems are split already) or separator= (split the "
            "mix here), not both"
        )
    source = float_input(audio, sr)
    x, rate = source.samples, source.sr
    if not 0 < tone_hz < rate / 2:
        raise ValueError(
            f"tone_hz must be in (0, {rate / 2}) Hz at {rate} Hz, got {tone_hz}"
        )
    words = as_words(spans)
    if widen_floor_db is None:
        widened = [(w.start, w.end) for w in words]
    else:
        widened = widen_spans(
            x, rate, words, floor_db=widen_floor_db, max_s=widen_max_s
        )
    out = x
    if words:
        out = remove_talk(
            x, widened, separator=separator or MUTE, margin=fade_s, sr=rate
        )
        removed = x - out  # the voice the mute took out (all of it, inside a word)
        gain = 10.0 ** (beep_db / 20.0)
        beeps = [
            (span, gain * _rms(removed, _frames((w.start, w.end), rate, len(x))))
            for span, w in zip(widened, words)
        ]
        tone = beep_track(len(x), rate, beeps, tone_hz=tone_hz, fade_s=fade_s)
        out = out + tone[:, None]
    if backing is not None:
        out = out + _fit_backing(backing, rate, len(x), x.shape[1])
    return deliver(
        source.restore(out),
        output,
        write=lambda _, path: source.as_audio(out).save(path, **save_kwargs),
        default_name=f"{source.stem}{OUTPUT_SUFFIX}",
    )


__all__ = [
    "bleep",
    "widen_spans",
    "beep_track",
    "as_words",
    "Word",
    "MUTE",
    "BEEP_TONE_HZ",
]
