"""Code-switched audio: per-language utterances, runs, and translation pairs.

A recording that alternates between two languages (a language lesson reading a
phrase then its translation, a bilingual interview, an interpreter's track)
defeats whole-file ASR: Whisper picks one language for the file and drops or
translates the other. What works is three steps, each a function here:

1. :func:`language_utterances`: split at silences (``find_segments``'s
   ``"silence"`` strategy), then per utterance a language id and a
   transcription forced to that language. Both are seams (``language_id=``,
   ``transcribe=``).
2. :func:`language_runs`: consecutive utterances of one language merged into
   labelled :class:`~mixing.audio.segmentation.Segment` runs, uncertain ones
   joining the run they sit in, junk (hallucinated text on sub-second words)
   dropped.
3. :func:`anchored_pairs`: pairs anchored on the *reliable* language. One
   language is usually identified near-certainly, the other is noisy on single
   words; taking the reliable language's runs as anchors and the speech
   *between* them as the other language recovers pairs that per-utterance
   labels lose.

The defaults of the seams are local and share one model:
:func:`whisper_language_id` (Whisper's own language id) and
:func:`whisper_transcriber`, both through ``mlx_whisper`` (Apple silicon);
without it, transcription falls back to :func:`scribed_transcriber` (the
fleet's ``scribed``, its default local backend). None is a mixing dependency;
each is imported on first use. The steps after the first are pure
(no audio, no model), so they can be re-tuned on stored utterances.

>>> utts = [Utterance(0.0, 1.0, {"zh": 0.9, "fr": 0.1}, "ni hao"),
...         Utterance(1.2, 2.0, {"zh": 0.0, "fr": 1.0}, "bonjour")]
>>> [(p.other.label, p.anchor.label) for p in anchored_pairs(utts, anchor="fr")]
[('zh', 'fr')]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from mixing.audio.segmentation import Segment, find_segments

#: ``(mono float samples, sample rate) -> {language: probability}``.
LanguageId = Callable[[np.ndarray, int], Mapping[str, float]]
#: ``(mono float samples, sample rate, language) -> text``.
Transcriber = Callable[[np.ndarray, int, str], str]

DFLT_LANGUAGES: tuple[str, ...] = ("fr", "zh")
#: Silence that ends an utterance (seconds) and how quiet it is (dBFS):
#: fine enough to split a phrase from its translation (measured on lessons).
DFLT_MIN_SILENCE_S = 0.35
DFLT_SILENCE_THRESH_DB = -40.0
#: Kept on each side of an utterance, so a word's onset and tail survive.
DFLT_PAD_S = 0.08
#: Shorter utterances are dropped (a click, a breath).
DFLT_MIN_UTTERANCE_S = 0.15
#: Whisper's input rate; utterances are handed to the seams at this rate.
WHISPER_SAMPLE_RATE = 16000
#: The model the starter measured with (Apple silicon, mlx).
DFLT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
#: Text a Whisper hallucinates on near-silence or a lone syllable (subtitle
#: credits, "like and subscribe" in its training captions). Extend with
#: ``functools.partial(is_junk, markers=HALLUCINATION_MARKERS + (...,))``.
HALLUCINATION_MARKERS: tuple[str, ...] = (
    "usz",
    "字幕",
    "Sous-titrage",
    "订阅",
    "点赞",
)
#: An utterance shorter than this whose language is uncertain is junk.
DFLT_SHORT_UNCERTAIN_S = 0.35
#: A language is decided when its probability reaches this ...
DFLT_MIN_PROBABILITY = 0.5
#: ... or when every other language's is at most this.
DFLT_MAX_OTHER_PROBABILITY = 0.1
#: The two fixed pair orders of :func:`anchored_pairs`, and the one it detects.
OTHER_THEN_ANCHOR = "other-then-anchor"
ANCHOR_THEN_OTHER = "anchor-then-other"
AUTO_ORDER = "auto"
PAIR_ORDERS: tuple[str, ...] = (OTHER_THEN_ANCHOR, ANCHOR_THEN_OTHER)
#: ``order="auto"`` costs, in seconds of pause inside a pair: switching order
#: between consecutive pairs ...
DFLT_SWITCH_COST_S = 1.0
#: ... and leaving an anchor or a stretch of speech unpaired.
DFLT_UNPAIRED_COST_S = 1.5


@dataclass(frozen=True)
class Utterance:
    """One silence-bounded stretch of speech: its span (seconds), the language
    id's ``probs`` and the transcription (forced to :attr:`language`)."""

    start: float
    end: float
    probs: Mapping[str, float] = field(default_factory=dict)
    text: str = ""

    @property
    def language(self) -> Optional[str]:
        """The most probable language, or None without probabilities."""
        return max(self.probs, key=self.probs.get) if self.probs else None

    @property
    def duration(self) -> float:
        """Length in seconds (``end - start``)."""
        return self.end - self.start

    def decided(
        self,
        *,
        min_probability: float = DFLT_MIN_PROBABILITY,
        max_other_probability: float = DFLT_MAX_OTHER_PROBABILITY,
    ) -> Optional[str]:
        """The language this utterance is, or None when it is ambiguous: the
        most probable one when it reaches ``min_probability``, else the one
        every other language is at most ``max_other_probability`` beside.
        Probabilities are the language id's own (a Whisper spreads mass over
        languages not asked about), not renormalized.

        >>> Utterance(0, 1, {"fr": 0.3, "zh": 0.05}).decided()
        'fr'
        >>> Utterance(0, 1, {"fr": 0.3, "zh": 0.3}).decided() is None
        True

        When every language is that weak, nothing stands out: ambiguous (#218).

        >>> Utterance(0, 1, {"zh": 0.02, "fr": 0.05}).decided() is None
        True
        """
        top = self.language
        if top is None:
            return None
        if self.probs[top] >= min_probability:
            return top
        others_weak = all(
            p <= max_other_probability for k, p in self.probs.items() if k != top
        )
        if others_weak and self.probs[top] > max_other_probability:
            return top
        return None


@dataclass(frozen=True)
class LanguagePair:
    """An anchor run (the reliably identified language) and the other
    language's speech paired with it. One side may be None: an anchor with no
    speech beside it, or other-language speech with no anchor to pair it with
    (before the first anchor or after the last one)."""

    anchor: Optional[Segment]
    other: Optional[Segment] = None

    @property
    def order(self) -> Optional[str]:
        """``"other-then-anchor"`` or ``"anchor-then-other"``, read from the
        spans (which side starts first; so ``order="auto"`` shows where it
        switched); None when a side is missing.

        >>> LanguagePair(Segment(1.0, 2.0, "fr"), Segment(0.0, 0.8, "zh")).order
        'other-then-anchor'
        """
        if self.anchor is None or self.other is None:
            return None
        if self.other.start < self.anchor.start:
            return OTHER_THEN_ANCHOR
        return ANCHOR_THEN_OTHER


# --- step 1: utterances ---------------------------------------------------------


def _mono(audio, sr: Optional[int]) -> tuple[np.ndarray, int]:
    from mixing.audio.samples import float_input
    from mixing.resample import resample

    fi = float_input(audio, sr)
    mono = fi.samples.mean(axis=1)
    return resample(mono, fi.sr, WHISPER_SAMPLE_RATE), WHISPER_SAMPLE_RATE


def _as_segment(samples: np.ndarray, sr: int):
    from pydub import AudioSegment

    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    return AudioSegment(ints.tobytes(), sample_width=2, frame_rate=sr, channels=1)


def language_utterances(
    audio,
    *,
    sr: Optional[int] = None,
    languages: Sequence[str] = DFLT_LANGUAGES,
    language_id: Optional[LanguageId] = None,
    transcribe: Optional[Transcriber] = None,
    min_silence_s: float = DFLT_MIN_SILENCE_S,
    silence_thresh_db: float = DFLT_SILENCE_THRESH_DB,
    pad_s: float = DFLT_PAD_S,
    min_utterance_s: float = DFLT_MIN_UTTERANCE_S,
) -> list[Utterance]:
    """``audio`` (a path, ``Audio``, pydub segment, or an array with ``sr``)
    split at silences into :class:`Utterance` s, each language-identified among
    ``languages`` and transcribed in its language.

    ``language_id`` defaults to :func:`whisper_language_id`, ``transcribe`` to
    :func:`default_transcriber`. A language in ``languages`` that the language
    id never names (``"zh-cn"`` where Whisper says ``"zh"``) is an error. :func:`anchored_pairs` needs no text: pass
    ``transcribe=lambda samples, sr, language: ""`` to skip it (then give
    :func:`language_runs` ``junk=None``, since its default junk filter drops
    utterances without text). Utterances reach the seams as mono float samples
    at :data:`WHISPER_SAMPLE_RATE`.
    """
    language_id = language_id or whisper_language_id()
    transcribe = transcribe or default_transcriber()
    samples, rate = _mono(audio, sr)
    total = len(samples) / rate
    spans = find_segments(
        _as_segment(samples, rate),
        strategy="silence",
        min_silence_len=min_silence_s,
        silence_thresh_db=silence_thresh_db,
    )
    out, named = [], set()
    for s in spans:
        a, b = max(0.0, s.start - pad_s), min(total, s.end + pad_s)
        if b - a < min_utterance_s:
            continue
        clip = samples[round(a * rate) : round(b * rate)]
        found = language_id(clip, rate)
        named.update(found)
        probs = {lang: float(found.get(lang, 0.0)) for lang in languages}
        best = max(probs, key=probs.get)
        out.append(
            Utterance(
                round(a, 3), round(b, 3), probs, transcribe(clip, rate, best).strip()
            )
        )
    unknown = [lang for lang in languages if lang not in named]
    if out and unknown:
        raise ValueError(
            f"the language id never named {unknown}: it uses codes like "
            f"{sorted(named)[:8]}"
        )
    return out


# --- step 2: runs ---------------------------------------------------------------


def is_junk(
    u: Utterance,
    *,
    markers: Sequence[str] = HALLUCINATION_MARKERS,
    short_s: float = DFLT_SHORT_UNCERTAIN_S,
    certain: float = DFLT_MIN_PROBABILITY,
) -> bool:
    """Whether ``u`` is a recogniser's hallucination rather than speech: no
    text, a known hallucination marker, or shorter than ``short_s`` with no
    language reaching ``certain``.

    >>> is_junk(Utterance(0.0, 0.2, {"fr": 0.4, "zh": 0.3}, "uh"))
    True
    >>> is_junk(Utterance(0.0, 0.3, {"fr": 0.0, "zh": 0.72}, "是"))
    False
    """
    if not u.text or any(m in u.text for m in markers):
        return True
    top = max(u.probs.values()) if u.probs else 0.0
    return u.duration < short_s and top < certain


def language_runs(
    utterances: Sequence[Utterance],
    *,
    min_probability: float = DFLT_MIN_PROBABILITY,
    max_other_probability: float = DFLT_MAX_OTHER_PROBABILITY,
    max_gap_s: float = 1.6,
    junk: Optional[Callable[[Utterance], bool]] = is_junk,
) -> list[Segment]:
    """Consecutive utterances of one language merged into runs: ``Segment`` s
    labelled with the language. An ambiguous utterance
    (:meth:`Utterance.decided` is None) joins the run it follows; a gap over
    ``max_gap_s`` starts a new run; ``junk`` (``None``: keep everything) is
    dropped first. Utterances are taken in time order.

    >>> utts = [Utterance(0, 1, {"zh": 1.0}, "a"), Utterance(1.2, 2, {"zh": 0.4, "fr": 0.3}, "b"),
    ...         Utterance(2.1, 3, {"fr": 1.0}, "c")]
    >>> [(r.start, r.end, r.label) for r in language_runs(utts, junk=None)]
    [(0, 2, 'zh'), (2.1, 3, 'fr')]
    """
    runs: list[Segment] = []
    for u in sorted(utterances, key=lambda u: u.start):
        if junk is not None and junk(u):
            continue
        lang = u.decided(
            min_probability=min_probability,
            max_other_probability=max_other_probability,
        )
        if lang is None:
            lang = runs[-1].label if runs else u.language
        if runs and runs[-1].label == lang and u.start - runs[-1].end <= max_gap_s:
            runs[-1] = Segment(runs[-1].start, u.end, lang)
        else:
            runs.append(Segment(u.start, u.end, lang))
    return runs


# --- step 3: pairs --------------------------------------------------------------


def anchored_pairs(
    utterances: Sequence[Utterance],
    *,
    anchor: str,
    other: Optional[str] = None,
    order: str = OTHER_THEN_ANCHOR,
    min_anchor_probability: float = DFLT_MIN_PROBABILITY,
    max_other_probability: float = DFLT_MAX_OTHER_PROBABILITY,
    max_gap_s: float = 0.6,
    min_other_s: float = 0.3,
    anchor_junk: Optional[Callable[[Utterance], bool]] = is_junk,
    other_junk: Optional[Callable[[Utterance], bool]] = None,
    switch_cost_s: float = DFLT_SWITCH_COST_S,
    unpaired_cost_s: float = DFLT_UNPAIRED_COST_S,
) -> list[LanguagePair]:
    """Pairs anchored on the reliably identified language ``anchor``.

    Utterances where ``anchor`` is at least ``min_anchor_probability`` and every
    other language at most ``max_other_probability`` are anchors, merged across
    gaps up to ``max_gap_s`` that hold no other speech. The speech between two
    anchor runs (``order``: before each anchor, ``"other-then-anchor"``, as a
    lesson reads a phrase then its translation; or after it,
    ``"anchor-then-other"``) is the ``other`` language (default: the most
    probable non-anchor language over the utterances), whatever its noisy
    per-utterance label said. Speech shorter than ``min_other_s`` leaves its
    anchor unpaired; speech no anchor claims (after the last anchor, or before
    the first) comes back as a pair with ``anchor=None``.

    ``order="auto"`` decides the order per pair, for a speaker who switches
    (reads "phrase, translation", then a list as "translation, phrase"). The
    utterances, in time order, are cut into consecutive blocks: a pair (other
    speech then anchors, or anchors then other speech), a lone anchor run, or
    lone other speech. The cut chosen is the one of least cost, found exactly
    by dynamic programming, where the cost, in seconds, is

    - every pause *inside* a block (a phrase and its translation sit closer
      than two pairs do; this also prices merging two anchors or two stretches
      of speech into one block),
    - plus ``switch_cost_s`` each time a pair's order differs from the previous
      pair's,
    - plus ``unpaired_cost_s`` for each block left unpaired (a lone anchor run,
      lone speech, or a pair whose speech is shorter than ``min_other_s``).

    Anchors still merge only across gaps up to ``max_gap_s`` with no speech
    between, but in ``auto`` merging is a choice, not a rule. On a recording
    that keeps one order, auto returns that order's pairs: switching leaves an
    anchor or a stretch of speech unpaired, which costs more than it saves,
    as long as a pair is not looser than the pause between pairs (a recording
    whose pairs are much looser than the pauses between them reads as the
    other order). On equal costs (to the microsecond) it keeps
    ``"other-then-anchor"``. A run of n utterances of one language costs
    O(n^2) work, so a long one-language stretch is slow (4000 in a row: about
    6 s); alternating speech is linear. Each pair's
    :attr:`LanguagePair.order` says which order it was read in.

    The two junk filters have different jobs: ``anchor_junk`` (default
    :func:`is_junk`) keeps a hallucinated transcript from becoming an anchor;
    ``other_junk`` (default None) would drop speech from the other side, which a
    hallucinated transcript does not prove absent. Utterances are taken in time
    order.

    >>> utts = [Utterance(0.0, 1.0, {"zh": 0.9}, "ni hao"),
    ...         Utterance(1.2, 2.0, {"fr": 1.0}, "bonjour"),
    ...         Utterance(3.0, 4.0, {"fr": 1.0}, "trente"),
    ...         Utterance(4.2, 5.0, {"zh": 0.9}, "san shi")]
    >>> [p.order for p in anchored_pairs(utts, anchor="fr", order="auto")]
    ['other-then-anchor', 'anchor-then-other']
    """
    if order not in (*PAIR_ORDERS, AUTO_ORDER):
        raise ValueError(f"order is one of {(*PAIR_ORDERS, AUTO_ORDER)}, got {order!r}")
    utts = sorted(utterances, key=lambda u: u.start)
    other = other or _other_language(utts, anchor)

    def is_anchor(u: Utterance) -> bool:
        if anchor_junk is not None and anchor_junk(u):
            return False
        rest = [p for lang, p in u.probs.items() if lang != anchor]
        return u.probs.get(anchor, 0.0) >= min_anchor_probability and all(
            p <= max_other_probability for p in rest
        )

    def as_other(inside: Sequence[Utterance]) -> Optional[Segment]:
        if not inside or inside[-1].end - inside[0].start < min_other_s:
            return None
        return Segment(inside[0].start, inside[-1].end, other)

    tokens = [
        (u, is_anchor(u))
        for u in utts
        if is_anchor(u) or other_junk is None or not other_junk(u)
    ]
    if order == AUTO_ORDER:
        blocks = _cheapest_blocks(
            tokens,
            max_gap_s=max_gap_s,
            min_other_s=min_other_s,
            switch_cost_s=switch_cost_s,
            unpaired_cost_s=unpaired_cost_s,
        )
        return [
            LanguagePair(
                Segment(a[0].start, a[-1].end, anchor) if a else None,
                as_other(o),
            )
            for a, o in blocks
            if a or as_other(o) is not None
        ]

    others = [u for u, is_a in tokens if not is_a]

    def speech_between(lo: float, hi: float) -> list[Utterance]:
        return [u for u in others if u.start >= lo and u.end <= hi]

    anchors: list[list[float]] = []
    for u in (u for u, is_a in tokens if is_a):
        if (
            anchors
            and u.start - anchors[-1][1] <= max_gap_s
            and not speech_between(anchors[-1][1], u.start)
        ):
            anchors[-1][1] = u.end
        else:
            anchors.append([u.start, u.end])

    pairs = []
    edges = [-np.inf] + [e for a in anchors for e in a] + [np.inf]
    for k, (a0, a1) in enumerate(anchors):
        lo, hi = (
            (edges[2 * k], a0) if order == OTHER_THEN_ANCHOR else (a1, edges[2 * k + 3])
        )
        pairs.append(
            LanguagePair(Segment(a0, a1, anchor), as_other(speech_between(lo, hi)))
        )
    # the speech no anchor claims: after the last anchor, or before the first
    lo, hi = (edges[-2], np.inf) if order == OTHER_THEN_ANCHOR else (-np.inf, edges[1])
    leftover = as_other(speech_between(lo, hi))
    if leftover is not None:
        if order == OTHER_THEN_ANCHOR:
            pairs.append(LanguagePair(None, leftover))
        else:
            pairs.insert(0, LanguagePair(None, leftover))
    return pairs


#: Costs are compared to the microsecond, so float rounding never breaks a tie.
_COST_DECIMALS = 6
#: A block of :func:`_cheapest_blocks`: its anchor utterances and its other ones.
_Block = tuple[list[Utterance], list[Utterance]]


def _cheapest_blocks(
    tokens: Sequence[tuple[Utterance, bool]],
    *,
    max_gap_s: float,
    min_other_s: float,
    switch_cost_s: float,
    unpaired_cost_s: float,
) -> list[_Block]:
    """The least-cost cut of ``(utterance, is_anchor)`` tokens into blocks,
    the ``order="auto"`` rule of :func:`anchored_pairs` (whose docstring states
    the cost). Dynamic programming over (end of the last block, order of the
    last pair); a block spans at most two runs of one kind, so the work grows
    with the token count times the longest two runs. Equal costs go to the cut
    with fewer ``"anchor-then-other"`` pairs."""
    n = len(tokens)
    kinds = [is_a for _, is_a in tokens]
    pauses = [max(0.0, tokens[k + 1][0].start - tokens[k][0].end) for k in range(n - 1)]

    def exact(seconds: float) -> float:
        return round(seconds, _COST_DECIMALS)  # equal costs compare equal

    def blocks_ending_at(j: int):
        """``(i, cost, order)`` for every block tokens[i:j], i from j - 1 down."""
        inside, flips, side_end, side_start = 0.0, 0, None, None
        for i in range(j - 1, -1, -1):
            if i < j - 1:
                if kinds[i] != kinds[i + 1]:
                    flips += 1
                    if flips > 1:
                        return
                elif kinds[i] and pauses[i] > max_gap_s:
                    return  # anchors merge only across a short gap
                inside += pauses[i]
            if not kinds[i]:
                side_end = side_end if side_end is not None else tokens[i][0].end
                side_start = tokens[i][0].start
            if flips == 0 or side_end - side_start < min_other_s:
                yield i, inside + unpaired_cost_s, None
            else:
                yield i, inside, ANCHOR_THEN_OTHER if kinds[i] else OTHER_THEN_ANCHOR

    # best[(j, last pair order)] = ((cost, anchor-then-other count), i, previous order)
    best: dict = {(0, None): ((0.0, 0), -1, None)}
    for j in range(1, n + 1):
        for i, cost, block_order in blocks_ending_at(j):
            for last in (None, *PAIR_ORDERS):
                if (i, last) not in best:
                    continue
                (prev_cost, prev_after), _, _ = best[(i, last)]
                switched = block_order and last and block_order != last
                score = (
                    exact(prev_cost + cost + (switch_cost_s if switched else 0.0)),
                    prev_after + (block_order == ANCHOR_THEN_OTHER),
                )
                state = (j, block_order or last)
                if state not in best or score < best[state][0]:
                    best[state] = (score, i, last)
    state = min((s for s in best if s[0] == n), key=lambda s: best[s][0])
    cuts = []
    while state[0] > 0:
        _, i, last = best[state]
        cuts.append((i, state[0]))
        state = (i, last)
    return [
        (
            [u for u, is_a in tokens[i:j] if is_a],
            [u for u, is_a in tokens[i:j] if not is_a],
        )
        for i, j in reversed(cuts)
    ]


def _other_language(utts: Sequence[Utterance], anchor: str) -> Optional[str]:
    totals: dict[str, float] = {}
    for u in utts:
        for lang, p in u.probs.items():
            if lang != anchor:
                totals[lang] = totals.get(lang, 0.0) + p
    return max(totals, key=totals.get) if totals else None


# --- the default seams ----------------------------------------------------------


def _mlx_missing(e: ModuleNotFoundError, what: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"{what} needs mlx_whisper (pip install mlx-whisper, Apple silicon); "
        f"or pass your own seam",
        name=e.name,
    )


@lru_cache(maxsize=None)
def _whisper_model(model: str):
    """The mlx Whisper ``model``, loaded once per process."""
    from huggingface_hub import snapshot_download
    from mlx_whisper.load_models import load_model

    return load_model(snapshot_download(model))


def whisper_language_id(model: str = DFLT_WHISPER_MODEL) -> LanguageId:
    """A :data:`LanguageId` from Whisper's own language detection, through
    ``mlx_whisper`` (Apple silicon; ``pip install mlx-whisper``). The model is
    loaded on the first call and kept for the process."""

    def language_id(samples: np.ndarray, sr: int) -> Mapping[str, float]:
        if sr != WHISPER_SAMPLE_RATE:
            raise ValueError(f"Whisper wants {WHISPER_SAMPLE_RATE} Hz, got {sr}")
        try:
            import mlx.core as mx
            from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
            from mlx_whisper.decoding import detect_language
        except ModuleNotFoundError as e:
            raise _mlx_missing(e, "whisper_language_id") from e
        m = _whisper_model(model)
        audio = pad_or_trim(mx.array(samples.astype(np.float32)), N_SAMPLES)
        mel = log_mel_spectrogram(audio, n_mels=m.dims.n_mels)
        _, probs = detect_language(m, mel[None].astype(mx.float16))
        return {k: float(v) for k, v in probs[0].items()}

    return language_id


def whisper_transcriber(model: str = DFLT_WHISPER_MODEL, **options) -> Transcriber:
    """A :data:`Transcriber` through ``mlx_whisper``, the same model as
    :func:`whisper_language_id` by default, forced to the utterance's language
    and not conditioned on earlier text (each utterance stands alone)."""

    def transcribe(samples: np.ndarray, sr: int, language: str) -> str:
        if sr != WHISPER_SAMPLE_RATE:
            raise ValueError(f"Whisper wants {WHISPER_SAMPLE_RATE} Hz, got {sr}")
        try:
            import mlx_whisper
        except ModuleNotFoundError as e:
            raise _mlx_missing(e, "whisper_transcriber") from e
        kwargs = {"condition_on_previous_text": False, **options}
        result = mlx_whisper.transcribe(
            samples.astype(np.float32),
            path_or_hf_repo=model,
            language=language,
            **kwargs,
        )
        return result["text"]

    return transcribe


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """Mono float samples as 16-bit WAV bytes (the rate travels with them)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(
            np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2").tobytes()
        )
    return buf.getvalue()


def scribed_transcriber(*, backend: Optional[str] = None, **options) -> Transcriber:
    """A :data:`Transcriber` through ``scribed`` (its default local backend, or
    ``backend``), forced to the utterance's language."""

    def transcribe(samples: np.ndarray, sr: int, language: str) -> str:
        try:
            import scribed
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "scribed_transcriber needs scribed (pip install scribed); "
                "or pass your own transcribe=",
                name=e.name,
            ) from e
        return scribed.transcribe_text(
            _wav_bytes(samples, sr), backend=backend, language=language, **options
        )

    return transcribe


def default_transcriber() -> Transcriber:
    """:func:`whisper_transcriber` when ``mlx_whisper`` is installed (the model
    the language id already loaded), else :func:`scribed_transcriber`."""
    import importlib.util

    if importlib.util.find_spec("mlx_whisper") is not None:
        return whisper_transcriber()
    return scribed_transcriber()


__all__ = [
    "Utterance",
    "LanguagePair",
    "LanguageId",
    "Transcriber",
    "language_utterances",
    "is_junk",
    "language_runs",
    "anchored_pairs",
    "whisper_language_id",
    "whisper_transcriber",
    "scribed_transcriber",
    "default_transcriber",
]
