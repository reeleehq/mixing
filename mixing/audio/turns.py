"""Process each turn of a recording separately and splice the results back.

``process_turns(audio, turns, process)`` is the combinator behind
:func:`mixing.audio.leveling.level_turns`, and is reusable for any
per-segment treatment (per-speaker EQ, denoising, ducking, ...):

- turns are ``(speaker, start_s, end_s)`` in **seconds**, whatever the
  ``Audio.time_unit``, or :class:`~mixing.audio.segmentation.Segment` s whose
  ``label`` is the speaker; all slicing is done in sample frames of
  ``audio._get_segment()``, so the unit cannot break it;
- each turn is cut **padded** by ``pad_s`` on both sides (clamped to the audio)
  so filters have settled by the time the turn itself starts;
- ``process(speaker, padded, core)`` gets that padded clip plus ``core``, the
  ``(start_s, end_s)`` of the turn inside it, and returns the processed padded
  clip;
- the combinator trims the turn back out and forces it to the **exact** frame
  count of the original turn (zero-padding or cutting), so the output duration
  equals the input's even if a stage returned shorter or longer audio;
- gaps between turns are copied untouched, and all pieces are joined in one
  ``b"".join`` (linear in the number of turns).

>>> from pydub import AudioSegment
>>> audio = Audio(AudioSegment.silent(1000, frame_rate=8000))
>>> out = process_turns(audio, [("a", 0.2, 0.5)], lambda s, a, core: a[: core[0]])
>>> out.duration
1.0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence, Union

from .audio_ops import Audio

if TYPE_CHECKING:
    from .segmentation import Segment

Turn = tuple[str, float, float]  # (speaker, start_s, end_s)
#: A turn as given: a ``Turn`` tuple or a ``Segment`` (``label`` = speaker).
TurnLike = Union[Turn, "Segment"]
Core = tuple[float, float]  # (start_s, end_s) of the turn inside its padded clip
TurnProcess = Callable[[str, Audio, Core], Audio]

DEFAULT_PAD_S = 0.5
#: Slack (seconds) allowed on turn bounds: float noise in a caller's timestamps.
TURN_BOUND_TOLERANCE_S = 1e-6


def as_turn(turn: TurnLike) -> Turn:
    """``turn`` as a ``(speaker, start_s, end_s)`` tuple; a ``Segment``'s
    ``label`` is the speaker (an unlabelled one is a ``ValueError``)."""
    if isinstance(turn, (tuple, list)):  # namedtuples, JSON lists
        speaker, start, end = turn
        return (speaker, start, end)
    if turn.label is None:
        raise ValueError(f"turn {turn!r} has no label to use as its speaker")
    return (turn.label, turn.start, turn.end)


def check_turns(
    turns: Sequence[TurnLike],
    duration: float,
    *,
    tolerance: float = TURN_BOUND_TOLERANCE_S,
) -> list[Turn]:
    """``turns`` as tuples sorted by start; raises ``ValueError`` if out of
    range or overlapping."""
    ordered = sorted(map(as_turn, turns), key=lambda t: t[1])
    previous_end = 0.0
    for speaker, start, end in ordered:
        if not 0 <= start < end <= duration + tolerance:
            raise ValueError(
                f"bad turn {(speaker, start, end)} for {duration:.2f}s audio"
            )
        if start < previous_end - tolerance:
            raise ValueError(f"turn {(speaker, start, end)} overlaps the previous one")
        previous_end = end
    return ordered


def conform_segment(segment, like):
    """``segment`` converted to ``like``'s frame rate, channels and sample width."""
    if segment.frame_rate != like.frame_rate:
        segment = segment.set_frame_rate(like.frame_rate)
    if segment.channels != like.channels:
        segment = segment.set_channels(like.channels)
    if segment.sample_width != like.sample_width:
        segment = segment.set_sample_width(like.sample_width)
    return segment


def process_turns(
    audio: Union[str, Path, Audio],
    turns: Sequence[TurnLike],
    process: TurnProcess,
    *,
    pad_s: float = DEFAULT_PAD_S,
) -> Audio:
    """Apply ``process`` to every turn of ``audio``; gaps stay untouched.

    Args:
        audio: the recording (path or :class:`Audio`).
        turns: ``[(speaker, start_s, end_s), ...]`` in seconds, non-overlapping
            (or ``Segment`` s, ``label`` being the speaker).
        process: ``process(speaker, padded, core) -> padded'``; ``padded`` is the
            turn with ``pad_s`` of context each side (a seconds-unit
            :class:`Audio`), ``core`` the turn's ``(start_s, end_s)`` within it.
            The result is trimmed back to ``core`` and forced to its exact
            length, so it may drift in length without shifting what follows.
        pad_s: context kept around each turn while processing.

    Returns:
        A new :class:`Audio` (same ``time_unit``, rate, channels, sample width
        and frame count as ``audio``'s segment).
    """
    audio = audio if isinstance(audio, Audio) else Audio(audio)
    segment = audio._get_segment()
    fs, width = segment.frame_rate, segment.frame_width
    raw = segment.raw_data
    n_frames = len(raw) // width
    ordered = check_turns(turns, n_frames / fs)
    pad_n = max(0, int(round(pad_s * fs)))

    pieces: list[bytes] = []
    cursor = 0  # frame index up to which the output is written
    for speaker, start, end in ordered:
        a = max(cursor, min(n_frames, int(round(start * fs))))
        b = max(a, min(n_frames, int(round(end * fs))))
        pieces.append(raw[cursor * width : a * width])
        if b > a:
            lo, hi = max(0, a - pad_n), min(n_frames, b + pad_n)
            padded = Audio(segment._spawn(raw[lo * width : hi * width]))
            out = process(speaker, padded, ((a - lo) / fs, (b - lo) / fs))
            out_raw = conform_segment(out._get_segment(), segment).raw_data
            piece = out_raw[(a - lo) * width : (b - lo) * width]
            piece += b"\0" * ((b - a) * width - len(piece))  # exact turn length
            pieces.append(piece)
        cursor = b
    pieces.append(raw[cursor * width :])
    return Audio(segment._spawn(b"".join(pieces)), time_unit=audio.time_unit)
