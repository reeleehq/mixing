"""Find the host's talk at the start and end of a song carved out of a radio show.

Songs cut from a radio tape (``find_segments(strategy="audioset", ...)``) often
still carry the announcer: an introduction spoken over the song's opening bars, or
a back-announce over the fade-out. :func:`find_talk_edges` says where that talk
is, so a caller can trim it, keep it, or record it as annotations, without
re-analysing the tape. Talk over music counts as talk.

How it works:

1. The classifier seam (the same one as the ``"audioset"`` strategy, see
   :mod:`mixing.audio.tagging`) scores P(speech) in short, densely overlapping
   windows (default: the AudioSet AST, 3 s windows every 0.5 s).
2. A window *seeds* talk if ``P(speech) >= talk_threshold + music_penalty *
   P(music)``, and a seeded region grows over neighbours down to ``talk_floor``
   (hysteresis). Sung vocals also read as speech to AudioSet (0.4-0.65 inside
   real singing, against 0.5-0.8 for a host over a music bed), so P(speech) alone
   cannot separate them; but singing comes with the full band, and a host talks
   over a ducked bed or a fade, so the bar rises with P(music).
3. The *song body* is the first (and, from the end, the last) run of non-talk
   windows lasting ``min_music_seconds``. Talk before the body is the intro, talk
   after it the outro. Talk inside the body (a sung line that reads as speech, a
   spoken bridge) is never an edge.
4. Window positions become times with ``edge_fraction`` (how much of a window
   must be talk for it to score as talk). With a word-timestamped transcript
   (``words=``), the intro end then snaps to the end of the last word starting
   in the ``word_snap_seconds`` before it (words just after it are usually the
   first lyrics), and the outro start to the first word starting within
   ``word_snap_seconds`` of it.

Without a transcript the edges are window-resolution estimates, typically within
a second. A transcript sharpens *where* an edge falls but not *whether* there is
talk: lyrics are transcribed too, so words alone cannot tell the host from the
singer, and the audio decides. It cannot rescue sung lines that read as talk
right next to the host (no long enough music between them): there the intro runs
on into the song.

>>> from mixing.audio.tagging import WindowProbabilities
>>> t = np.arange(0.0, 58.0, 0.5)              # 3 s windows every 0.5 s
>>> speech = np.where(t < 8, 0.8, 0.1)         # the host talks for the first 8 s
>>> probs = WindowProbabilities(t, t + 3, speech, 1 - speech)
>>> edges = talk_edges_from_probs(probs, start=0.0, end=60.0)
>>> edges.intro.start, float(edges.intro.end), edges.outro
(0.0, 8.5, None)
>>> float(edges.clean_start), edges.clean_end
(8.5, 60.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from .audio_util import AudioSource, _normalize_audio_source
from .segmentation import Segment, _label_runs, _mono_float32
from .tagging import (
    AST_SAMPLE_RATE,
    AudioClassifier,
    WindowProbabilities,
    ast_audioset_classifier,
)

#: Window length (seconds) of the default classifier: short, for fine edges.
TALK_WINDOW_SECONDS = 3.0
#: Hop (seconds) between windows of the default classifier.
TALK_HOP_SECONDS = 0.5
#: A window seeds a talk region if P(speech) reaches this (plus the music penalty).
DEFAULT_TALK_THRESHOLD = 0.45
#: The seed bar rises by this much per unit of P(music): the louder the music, the
#: more speech evidence a window needs (sung vocals come with full-band music).
DEFAULT_MUSIC_PENALTY = 0.4
#: Talk already under way where the song was cut (a region touching its start or
#: end) seeds at this P(speech), whatever the music: a host talking over the cut.
DEFAULT_EDGE_THRESHOLD = 0.6
#: A talk region extends over neighbouring windows down to this P(speech).
DEFAULT_TALK_FLOOR = 0.4
#: The song body is the first run of non-talk windows at least this long (seconds).
DEFAULT_MIN_MUSIC_SECONDS = 8.0
#: Intro and outro are searched only this far (seconds) from the song's edges.
DEFAULT_MAX_EDGE_SECONDS = 60.0
#: Fraction of a window that must hold talk for it to score as talk; turns window
#: positions into edge times (calibrated on hand-labelled radio-show songs).
DEFAULT_EDGE_FRACTION = 0.25
#: An edge snaps to a transcript word starting within this many seconds of it.
DEFAULT_WORD_SNAP_SECONDS = 1.5
#: Longest duration (seconds) credited to a transcript word. Caption words often
#: "end" where the next word starts, which after the host's last word is seconds away
#: (a multi-word item, such as a cue, is credited this much per word).
MAX_WORD_SECONDS = 0.6
#: Label of :attr:`TalkEdges.clean`.
CLEAN_LABEL = "music"


@dataclass(frozen=True)
class TalkEdges:
    """Where the host talks at the edges of one song, in seconds.

    Attributes:
        start: Start of the analysed song (the segment start, or 0).
        end: End of the analysed song.
        intro: The talk at the start, from its first to its last talk moment
            (stings or a few seconds of music in between included), or ``None``.
        outro: The talk at the end, or ``None``.
        clean_start: Where the music proper starts (``intro.end``, else ``start``).
        clean_end: Where it ends (``outro.start``, else ``end``).
        snapped_to_words: True if a transcript refined at least one edge.
    """

    start: float
    end: float
    intro: Segment | None
    outro: Segment | None
    clean_start: float
    clean_end: float
    snapped_to_words: bool = False

    @property
    def clean(self) -> Segment:
        """The song without its intro and outro talk, as a ``Segment``."""
        return Segment(self.clean_start, self.clean_end, label=CLEAN_LABEL)


def _talk_flags(
    speech: np.ndarray,
    music: np.ndarray,
    *,
    threshold: float,
    music_penalty: float,
    edge_threshold: float,
    floor: float,
) -> np.ndarray:
    """Hysteresis: runs of ``speech >= floor`` holding a seed window.

    A seed has ``speech >= threshold + music_penalty * music``; in a run touching
    the first or last window, ``speech >= edge_threshold`` is enough.
    """
    seeds = speech >= threshold + music_penalty * music
    edge_seeds = speech >= edge_threshold
    flags = np.zeros(len(speech), dtype=bool)
    for above, a, b in _label_runs(speech >= floor):
        touches_edge = a == 0 or b == len(speech)
        if above and (seeds[a:b].any() or (touches_edge and edge_seeds[a:b].any())):
            flags[a:b] = True
    return flags


def _body_bounds(
    flags: np.ndarray, starts: np.ndarray, ends: np.ndarray, *, min_music_seconds: float
) -> tuple[int, int] | None:
    """Indices of the first and last window of the song body (long non-talk runs)."""
    long_runs = [
        (a, b)
        for is_talk, a, b in _label_runs(flags)
        if not is_talk
        and starts[b - 1] - starts[a] + (ends[a] - starts[a]) / 2 >= min_music_seconds
    ]
    if not long_runs:
        return None
    return long_runs[0][0], long_runs[-1][1] - 1


def talk_edges_from_probs(
    probs: WindowProbabilities,
    *,
    start: float,
    end: float,
    talk_threshold: float = DEFAULT_TALK_THRESHOLD,
    music_penalty: float = DEFAULT_MUSIC_PENALTY,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    talk_floor: float = DEFAULT_TALK_FLOOR,
    min_music_seconds: float = DEFAULT_MIN_MUSIC_SECONDS,
    max_edge_seconds: float = DEFAULT_MAX_EDGE_SECONDS,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
    talk_label: str = "talk",
) -> TalkEdges:
    """The audio half of :func:`find_talk_edges`: edges from window probabilities.

    ``probs`` must be in the same timeline as ``start``/``end`` (seconds) and its
    windows should cover the song: talk where no window reaches is not seen. See
    :func:`find_talk_edges` for the arguments.
    """
    start, end = float(start), float(end)
    if not end > start:
        raise ValueError(f"Need start < end (seconds); got start={start}, end={end}.")
    if len(probs) == 0:
        return TalkEdges(start, end, None, None, start, end)
    starts, ends = np.asarray(probs.starts, float), np.asarray(probs.ends, float)
    if ends[-1] <= start or starts[0] >= end:
        raise ValueError(
            f"The windows ({starts[0]:.2f}-{ends[-1]:.2f} s) do not overlap the song "
            f"({start:.2f}-{end:.2f} s): are probs and start/end in the same timeline?"
        )
    flags = _talk_flags(
        np.asarray(probs.speech),
        np.asarray(probs.music),
        threshold=talk_threshold,
        music_penalty=music_penalty,
        edge_threshold=edge_threshold,
        floor=talk_floor,
    )
    body = _body_bounds(flags, starts, ends, min_music_seconds=min_music_seconds)
    if body is None:  # talk (or too little music to call a song) all the way
        talk = Segment(start, end, label=talk_label) if flags.any() else None
        if talk is None:
            return TalkEdges(start, end, None, None, start, end)
        return TalkEdges(start, end, talk, None, end, end)
    first_body, last_body = body
    width = ends - starts
    hop = float(np.median(np.diff(starts))) if len(starts) > 1 else 0.0

    def talk_end(i):  # last talk window i: talk ends ~edge_fraction into it
        return starts[i] + edge_fraction * width[i] + hop / 2

    def talk_start(i):  # first talk window i
        return ends[i] - edge_fraction * width[i] - hop / 2

    def clamp(t):
        return float(min(max(t, start), end))

    # A talk run that starts within max_edge_seconds of an edge is kept whole.
    runs = [(a, b) for is_talk, a, b in _label_runs(flags) if is_talk]
    head = [
        (a, b)
        for a, b in runs
        if b <= first_body and starts[a] < start + max_edge_seconds
    ]
    tail = [
        (a, b)
        for a, b in runs
        if a > last_body and ends[b - 1] > end - max_edge_seconds
    ]
    last = len(flags) - 1
    intro = outro = None
    if head:
        a = start if head[0][0] == 0 else clamp(talk_start(head[0][0]))
        b = clamp(talk_end(head[-1][1] - 1))
        intro = Segment(a, b, label=talk_label) if b > a else None
    if tail:
        a = clamp(talk_start(tail[0][0]))
        reaches_end = tail[-1][1] - 1 == last and ends[last] >= end - width[last]
        b = end if reaches_end else clamp(talk_end(tail[-1][1] - 1))
        outro = Segment(a, b, label=talk_label) if b > a else None
    clean_start = intro.end if intro else start
    clean_end = max(outro.start if outro else end, clean_start)
    return TalkEdges(start, end, intro, outro, clean_start, clean_end)


def _word_spans(words: Iterable[Any]) -> list[tuple[float, float]]:
    """``(start, end)`` of each transcript item: Scribe, ``parse_vtt_words`` or
    Whisper (``"word"`` key) mappings, or cue-like objects.

    An item is credited at most ``MAX_WORD_SECONDS`` per word of its text.
    """
    spans = []
    for w in words:
        if isinstance(w, Mapping):
            if w.get("type", "word") != "word":
                continue  # Scribe "spacing" and "audio_event" tokens
            text = w.get("text", w.get("word", ""))  # Scribe / Whisper
            s, e = w.get("start"), w.get("end")
        else:
            s, e = getattr(w, "start", None), getattr(w, "end", None)
            text = getattr(w, "text", "")
        n_words = len(str(text).split())
        if s is None or not n_words:
            continue
        s, longest = float(s), n_words * MAX_WORD_SECONDS
        e = s + longest if e is None else min(float(e), s + longest)
        spans.append((s, max(s, e)))
    return sorted(spans)


def _snap_to_words(
    edges: TalkEdges, spans: list[tuple[float, float]], *, radius: float
) -> TalkEdges:
    """Intro end -> end of the last word starting in the ``radius`` before it;
    outro start -> the first word starting within ``radius`` of it."""
    intro, outro, snapped = edges.intro, edges.outro, False
    if intro is not None and intro.end < edges.end:  # all-talk songs stay all talk
        ends = [e for s, e in spans if intro.end - radius <= s <= intro.end]
        if ends:
            end = min(max(ends[-1], intro.start), edges.end)
            intro = Segment(intro.start, end, label=intro.label)
            snapped = True
    if outro is not None:
        starts = [
            s for s, _ in spans if abs(s - outro.start) <= radius and s < outro.end
        ]
        if starts:
            outro = Segment(max(starts[0], edges.start), outro.end, label=outro.label)
            snapped = True
    clean_start = intro.end if intro else edges.start
    clean_end = max(outro.start if outro else edges.end, clean_start)
    return TalkEdges(
        edges.start, edges.end, intro, outro, clean_start, clean_end, snapped
    )


def find_talk_edges(
    audio: AudioSource,
    segment: Segment | tuple[float, float] | None = None,
    *,
    words: Iterable[Any] | None = None,
    classifier: AudioClassifier | None = None,
    sample_rate: int | None = AST_SAMPLE_RATE,
    talk_threshold: float = DEFAULT_TALK_THRESHOLD,
    music_penalty: float = DEFAULT_MUSIC_PENALTY,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    talk_floor: float = DEFAULT_TALK_FLOOR,
    min_music_seconds: float = DEFAULT_MIN_MUSIC_SECONDS,
    max_edge_seconds: float = DEFAULT_MAX_EDGE_SECONDS,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
    word_snap_seconds: float = DEFAULT_WORD_SNAP_SECONDS,
    talk_label: str = "talk",
) -> TalkEdges:
    """Find the host's talk at the start and end of one song (talk over music counts).

    Args:
        audio: The song (a file path, array or pydub ``AudioSegment``), or the
            whole tape when ``segment`` is given.
        segment: ``Segment`` or ``(start, end)`` seconds of the song inside
            ``audio`` (e.g. a ``"song"`` from ``find_segments(...,
            strategy="audioset")``). ``None`` analyses all of ``audio``. Returned
            times are always in ``audio``'s timeline.
        words: Optional word-timestamped transcript in ``audio``'s timeline:
            dicts with ``"start"`` (and ``"end"``) and ``"text"`` (Scribe's
            ``transcribe(...)["words"]``, :func:`mixing.srt.parse_vtt_words`) or
            ``"word"`` (Whisper)
            (shift caption times if ``audio`` is a carved song), or cue-like
            objects with ``.start``/``.end``. Edges snap to the announcer's last
            intro word and first outro word. Without it, edges are window-resolution
            estimates (about a second either way).
        classifier: The seam: any ``(samples, sr) -> WindowProbabilities``. ``None``
            uses :func:`~mixing.audio.tagging.ast_audioset_classifier` with
            ``TALK_WINDOW_SECONDS`` windows every ``TALK_HOP_SECONDS`` (needs
            ``pip install 'mixing[ml]'``). Only P(speech) and P(music) are used;
            its windows should cover the whole song.
        sample_rate: Rate the audio is resampled to (anti-aliased) before
            classifying; ``None`` keeps the native rate.
        talk_threshold: P(speech) that seeds a talk region where there is no
            music. Raise it (or ``music_penalty``) if sung openings come out as
            intro talk; lower it if a host over a loud bed is missed.
        music_penalty: How much the seed bar rises per unit of P(music).
        edge_threshold: P(speech) that seeds talk already under way where the song
            was cut (a talk region touching its first or last window), whatever
            the music.
        talk_floor: P(speech) down to which a talk region extends (hysteresis).
        min_music_seconds: Shortest non-talk run (seconds) taken as the song body;
            shorter music between talk (a sting, a pause) stays inside the intro.
        max_edge_seconds: How far (seconds) from either end a talk region may
            start to count as intro/outro; a region that does is kept whole.
        edge_fraction: Fraction of a window that must be talk for it to score as
            talk; converts window positions to times. Retune it with a classifier
            whose window length differs.
        word_snap_seconds: How far (seconds) an edge may move to reach a word:
            back from the audio's intro end, either way from its outro start.
        talk_label: Label of the ``intro``/``outro`` segments.

    Returns:
        A :class:`TalkEdges`: ``intro`` and ``outro`` (``Segment`` or ``None``),
        and ``clean_start``/``clean_end`` of the music proper. A song with no
        long enough music run comes back as all intro, with an empty clean span.
    """
    audio_seg = _normalize_audio_source(audio, target_type="AudioSegment")
    total = len(audio_seg) / 1000.0
    if segment is None:
        start, end = 0.0, total
    else:
        start, end = (
            (segment.start, segment.end) if hasattr(segment, "start") else segment
        )
        start, end = max(0.0, float(start)), min(total, float(end))
        if end <= start:
            raise ValueError(
                f"Empty segment ({start}, {end}) for audio of {total:.2f} s."
            )
        audio_seg = audio_seg[int(round(start * 1000)) : int(round(end * 1000))]
    samples, sr = _mono_float32(audio_seg, sample_rate=sample_rate)
    if len(samples) == 0:
        return TalkEdges(start, end, None, None, start, end)
    classify = classifier or ast_audioset_classifier(
        window_seconds=TALK_WINDOW_SECONDS, hop_seconds=TALK_HOP_SECONDS
    )
    p = classify(samples, sr)
    probs = WindowProbabilities(p.starts + start, p.ends + start, p.speech, p.music)
    edges = talk_edges_from_probs(
        probs,
        start=start,
        end=end,
        talk_threshold=talk_threshold,
        music_penalty=music_penalty,
        edge_threshold=edge_threshold,
        talk_floor=talk_floor,
        min_music_seconds=min_music_seconds,
        max_edge_seconds=max_edge_seconds,
        edge_fraction=edge_fraction,
        talk_label=talk_label,
    )
    if words is None:
        return edges
    spans = [(s, e) for s, e in _word_spans(words) if start <= s < end]
    return _snap_to_words(edges, spans, radius=word_snap_seconds)
