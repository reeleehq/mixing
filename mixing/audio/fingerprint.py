"""Does this clip contain the song? Landmark fingerprints, an independent second opinion.

A small numpy implementation of the landmark-hash method (Wang's Shazam paper [1],
Ellis's audfprint [2]; dejavu [3] is the same algorithm): the strongest local maxima
of a spectrogram form a *constellation*; each peak is paired with a few later peaks
in a target zone, and each pair hashes to ``(f1, f2, dt)``. A clip that contains the
song shares many hashes with it, and -- the part chance cannot fake -- those shared
hashes agree on **one** time offset.

It measures, from the histogram of ``t_song - t_clip`` over every shared hash:

- ``votes`` (**K**): the hashes agreeing on the best offset (within
  ``offset_tolerance_s``);
- ``rho`` (**rho**): ``K / shared``, the temporally consistent fraction of the
  shared hashes (chance stays under ~1 % [2]);
- ``ratio`` (**R**): ``K / K2``, ``K2`` the best rival offset more than
  ``rival_exclusion_s`` away (a rival of zero counts as one, so R is finite). A
  looped bar shares the clip's hashes with every repetition: K is high, R is ~1.

and calls one of THREE verdicts (:func:`song_verdict`): ``"contains"`` (K >= 12,
rho >= 0.05, R >= 2), ``"absent"`` (K <= 5, on a clip with enough landmarks to have
shown the song), else ``"undecided"`` -- a third outcome, never a pass.
:class:`SongMatch` has no truth value, so "undecided" cannot slip through an ``if``.

**The thresholds are uncalibrated.** They are the starting values of the
research behind mixing#55 (audfprint's "more than 5 or 6 consistently-timed
hashes" [2], tightened). Wang's procedure is to collect the score distribution
on *known non-matching* clips (bar-periodic aliases included) and pick the
threshold that meets a target false-positive rate; do that before letting this
vote. Every threshold is a keyword of :func:`contains_song` and of
:func:`song_verdict`, so stored measurements can be re-judged without rehashing.

Two choices beyond the textbook recipe, both measured on synthetic cuts (see
:func:`_peaks` and :func:`contains_song`): the constellation's density cap is
*local*, so a clip keeps the peaks its song had at the same place (a global or
block cap lost most of them under a low-pass), and the clip is tried at
``query_shifts`` sub-hop phases, the best one reported.

Hash a song once and reuse it: :func:`landmarks` returns a :class:`Landmarks`
index, which :func:`contains_song` accepts in place of the song's audio (the clip is
then hashed with the index's own :class:`LandmarkParams`, so the two always agree).

>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> song = rng.standard_normal(11025 * 8) * 0.1
>>> index = landmarks(song, sr=11025)
>>> match = contains_song(song[11025 * 2 : 11025 * 6], index, sr=11025)
>>> match.verdict, round(match.offset_s, 1)
('contains', 2.0)

References:
    1. Wang A. An industrial-strength audio search algorithm. ISMIR 2003.
    2. Ellis DPW. audfprint: landmark-based audio fingerprinting.
       https://www.ee.columbia.edu/~dpwe/LabROSA/matlab/audfprint/
    3. Drevo W. dejavu. https://github.com/worldveil/dejavu
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from ._pcm import resample, segment_samples

__all__ = [
    "Landmarks",
    "LandmarkParams",
    "SongMatch",
    "SongVerdict",
    "landmarks",
    "contains_song",
    "song_verdict",
    "CONTAINS",
    "ABSENT",
    "UNDECIDED",
    "SONG_VERDICTS",
]

# --- the three verdicts ------------------------------------------------------

SongVerdict = Literal["contains", "absent", "undecided"]
#: The song is in the clip: enough hashes agree on one offset, and it beats every rival.
CONTAINS: SongVerdict = "contains"
#: The clip had enough landmarks to show the song, and almost none agreed.
ABSENT: SongVerdict = "absent"
#: Could not decide -- too little evidence either way. Never a pass.
UNDECIDED: SongVerdict = "undecided"
SONG_VERDICTS: tuple[SongVerdict, ...] = (CONTAINS, ABSENT, UNDECIDED)

# --- hashing parameters (the research's recipe: 11.025 kHz, STFT 1024/256) ----

#: Analysis rate both clip and song are resampled to (Hz). Half of 22.05 kHz: the
#: band above ~5.5 kHz adds little a phone microphone keeps.
DEFAULT_ANALYSIS_SAMPLE_RATE = 11025
#: STFT window (samples): ~93 ms at the analysis rate, ~10.8 Hz bins.
DEFAULT_N_FFT = 1024
#: STFT hop (samples): ~23 ms -- the time resolution of every landmark and offset.
DEFAULT_HOP_LENGTH = 256
#: Full width of the local-maximum neighbourhood in frequency (Hz).
DEFAULT_PEAK_NEIGHBORHOOD_HZ = 200.0
#: Full width of the local-maximum neighbourhood in time (s).
DEFAULT_PEAK_NEIGHBORHOOD_S = 0.1
#: Size (time, s) of the local mean a peak's prominence is measured against.
DEFAULT_PROMINENCE_WINDOW_S = 1.0
#: Size (frequency, Hz) of that local mean: a smooth EQ or gain moves both alike.
DEFAULT_PROMINENCE_WINDOW_HZ = 1000.0
#: Constellation density kept (peaks per second): the top of the research's
#: 20-30. Measured on a low-passed, noisy clip, 25 kept too few of the peaks the
#: clip shares with its song (min K 7 over 8 cuts) where 30 kept enough (min K 17).
DEFAULT_PEAKS_PER_SECOND = 30.0
#: The density cap is local: a peak survives if fewer than
#: ``peaks_per_second * peak_density_window_s`` more prominent peaks lie within
#: half this window of it (s). Being local, a clip and the song it was cut from
#: keep the same peaks -- a cap over fixed blocks or the whole signal would not.
DEFAULT_PEAK_DENSITY_WINDOW_S = 1.0
#: Absolute floor (dB re a full-scale sine) a peak must clear: a near-silent
#: stretch yields no landmarks (its hiss would otherwise rank by prominence).
DEFAULT_PEAK_FLOOR_DB = -70.0
#: Pairs formed per anchor peak (the research's fan-out of ~5).
DEFAULT_FAN_OUT = 5
#: Target zone length after the anchor (s): pairs span at most this long.
DEFAULT_TARGET_ZONE_S = 2.0
#: Target zone half-height (Hz): a pair's two peaks are at most this far apart in
#: frequency (~audfprint's +-31 bins of 43 Hz).
DEFAULT_TARGET_ZONE_HZ = 1300.0
#: A pair must span time: same-frame pairs are chord partials, symmetric and doubled.
_MIN_PAIR_DT_FRAMES = 1

# --- matching parameters and the starting (uncalibrated) thresholds ----------

#: Sub-hop phases a clip is hashed at, each scored alone (audfprint's ``--shifts``).
DEFAULT_QUERY_SHIFTS = 4

#: Hashes within this of the best offset agree with it (s): +- one hop, the jitter
#: of a peak's frame when the clip does not start on the song's hop grid.
DEFAULT_OFFSET_TOLERANCE_S = 0.025
#: A rival offset is the best one more than this from the winner (s).
DEFAULT_RIVAL_EXCLUSION_S = 0.5
#: K at or above which the song may be called present (research: start at 12).
DEFAULT_MIN_VOTES = 12
#: rho at or above which the song may be called present (chance is < 1 %).
DEFAULT_MIN_RHO = 0.05
#: R at or above which the winning offset clearly beats every rival.
DEFAULT_MIN_RATIO = 2.0
#: K at or below which the song may be called absent (audfprint: > 5-6 is a match).
DEFAULT_MAX_ABSENT_VOTES = 5
#: Clip hashes needed before "absent" may be called: a present song keeping only
#: 5 % of its hashes would still clear DEFAULT_MIN_VOTES (12 / 0.05 = 240).
DEFAULT_MIN_QUERY_HASHES = 240

_EPS = 1e-12  # keeps log10 finite on digital silence


@dataclass(frozen=True)
class LandmarkParams:
    """How a signal is turned into landmarks. Clip and song must share one.

    Every default is the named module constant of the same name
    (``DEFAULT_<FIELD>``; ``sample_rate`` is ``DEFAULT_ANALYSIS_SAMPLE_RATE``). Override with ``LandmarkParams(fan_out=8)`` or
    ``dataclasses.replace(params, ...)``.
    """

    sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    peak_neighborhood_hz: float = DEFAULT_PEAK_NEIGHBORHOOD_HZ
    peak_neighborhood_s: float = DEFAULT_PEAK_NEIGHBORHOOD_S
    prominence_window_s: float = DEFAULT_PROMINENCE_WINDOW_S
    prominence_window_hz: float = DEFAULT_PROMINENCE_WINDOW_HZ
    peaks_per_second: float = DEFAULT_PEAKS_PER_SECOND
    peak_density_window_s: float = DEFAULT_PEAK_DENSITY_WINDOW_S
    peak_floor_db: float = DEFAULT_PEAK_FLOOR_DB
    fan_out: int = DEFAULT_FAN_OUT
    target_zone_s: float = DEFAULT_TARGET_ZONE_S
    target_zone_hz: float = DEFAULT_TARGET_ZONE_HZ

    @property
    def hop_s(self) -> float:
        """Seconds per STFT frame."""
        return self.hop_length / self.sample_rate

    @property
    def bin_hz(self) -> float:
        """Hz per STFT bin."""
        return self.sample_rate / self.n_fft

    def frames(self, seconds: float) -> int:
        """``seconds`` as a whole number of frames (at least one)."""
        return max(1, round(seconds / self.hop_s))

    def bins(self, hz: float) -> int:
        """``hz`` as a whole number of bins (at least one)."""
        return max(1, round(hz / self.bin_hz))


DEFAULT_LANDMARK_PARAMS = LandmarkParams()


@dataclass(frozen=True, eq=False)
class Landmarks:
    """A signal's landmark hashes and the frame each one's anchor sits on.

    Hash a song once and pass this to :func:`contains_song` for every clip.

    Attributes:
        hashes: ``int64`` hash of each ``(f1, f2, dt)`` pair.
        frames: ``int64`` anchor frame of each hash (``frames * params.hop_s`` s).
        params: the :class:`LandmarkParams` they were computed with.
        duration_s: length of the analysed signal (s).
    """

    hashes: np.ndarray
    frames: np.ndarray
    params: LandmarkParams = DEFAULT_LANDMARK_PARAMS
    duration_s: float = 0.0

    def __len__(self) -> int:
        return len(self.hashes)

    @cached_property
    def _sorted(self) -> tuple[np.ndarray, np.ndarray]:
        """``(sorted hashes, their frames)`` -- the lookup table, built on first match."""
        order = np.argsort(self.hashes, kind="stable")
        return self.hashes[order], self.frames[order]


@dataclass(frozen=True)
class SongMatch:
    """Whether a clip contains a song, with the evidence the verdict rests on.

    Deliberately has **no truth value** (``bool(match)`` raises): compare
    ``match.verdict == "contains"`` so "undecided" is never read as a pass.

    Attributes:
        verdict: ``"contains"``, ``"absent"`` or ``"undecided"`` (:data:`SONG_VERDICTS`).
        votes: K, the shared hashes agreeing on ``offset_s``.
        rival_votes: K2, the best count more than ``rival_exclusion_s`` from it.
        shared: every (clip hash, song hash) match, consistent or not.
        rho: ``votes / shared`` (0 when nothing is shared).
        ratio: R, ``votes / max(rival_votes, 1)``.
        offset_s: song time at the clip's t=0 (``t_song - t_clip``) for the best
            offset; ``None`` when nothing is shared. Only meaningful with
            ``"contains"``.
        query_hashes: the clip's own hash count, at the reported phase (the
            evidence it could offer).
        reason: one line on why this verdict.
    """

    verdict: SongVerdict
    votes: int
    rival_votes: int
    shared: int
    rho: float
    ratio: float
    offset_s: Optional[float]
    query_hashes: int
    reason: str = field(default="", compare=False)

    def __bool__(self) -> bool:
        raise TypeError(
            "SongMatch has no truth value: 'undecided' is not a pass. "
            f"Compare match.verdict == {CONTAINS!r} (verdict is {self.verdict!r})."
        )

    def to_dict(self) -> dict:
        """Plain-JSON form (for a record that stores the measurement)."""
        return asdict(self)


# --- input -------------------------------------------------------------------

AudioInput = Union[
    str, Path, np.ndarray, object
]  # path, array (+ sr), Audio, AudioSegment


def _mono_at(audio: AudioInput, *, sr: Optional[int], target: int) -> np.ndarray:
    """``audio`` as mono float samples in [-1, 1] at ``target`` Hz (anti-aliased)."""
    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError("an array needs its sample rate: pass sr=")
        data = np.asarray(audio, dtype=np.float64)
        if data.ndim > 2:
            raise ValueError(
                f"expected (n,) or (n, channels) samples, got {data.shape}"
            )
        mono = data.mean(axis=1) if data.ndim == 2 else data
        return resample(mono, int(sr), target)
    if sr is not None:
        raise ValueError("sr= is only for arrays; files and Audio carry their own rate")
    if hasattr(audio, "_get_segment"):  # mixing.audio.Audio (incl. a slice of one)
        segment = audio._get_segment()
    elif isinstance(audio, (str, Path)):
        from ..util import require_package

        segment = require_package("pydub").AudioSegment.from_file(str(audio))
    elif hasattr(audio, "get_array_of_samples"):  # a pydub AudioSegment
        segment = audio
    else:
        raise TypeError(
            f"unsupported audio {type(audio).__name__}: pass a path, an Audio, "
            "an AudioSegment, or an array with sr="
        )
    mono = segment_samples(segment).mean(axis=1)
    return resample(mono, segment.frame_rate, target)


# --- landmarks ---------------------------------------------------------------


def _spectrogram_db(y: np.ndarray, params: LandmarkParams) -> np.ndarray:
    """``(frames, bins)`` magnitude in dB re a full-scale sine (Hann, no centering)."""
    n_fft, hop = params.n_fft, params.hop_length
    if len(y) < n_fft:
        return np.empty((0, n_fft // 2 + 1))
    window = np.hanning(n_fft)
    framed = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    magnitude = np.abs(np.fft.rfft(framed * window, axis=1))
    full_scale_sine = window.sum() / 2
    return 20 * np.log10(magnitude / full_scale_sine + _EPS)


def _odd(n: int) -> int:
    """``n`` rounded up to odd, so a filter window has a centre."""
    return n + 1 - n % 2


def _peaks(
    spec_db: np.ndarray, params: LandmarkParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, bins)`` of the constellation, sorted by time then frequency.

    A peak is a local maximum of the spectrogram above ``peak_floor_db``, ranked
    by its *prominence* (its level minus the local mean level around it), and kept
    when it is among the ``peaks_per_second`` most prominent of its neighbourhood
    (:func:`_dense_enough`). Everything is local, so a gain change or a smooth EQ
    -- a muffled concert recording's low-pass -- leaves the choice alone, and a
    clip keeps the peaks the song had at the same place.

    (Whitening by each bin's median over the whole signal, the obvious way to
    remove spectral tilt, was measured to keep only ~45 % of the peaks a clip
    shares with its song: the clip's medians are not the song's.)
    """
    if spec_db.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    from scipy.ndimage import maximum_filter, uniform_filter

    neighborhood = (
        _odd(params.frames(params.peak_neighborhood_s)),
        _odd(params.bins(params.peak_neighborhood_hz)),
    )
    is_peak = (
        spec_db == maximum_filter(spec_db, size=neighborhood, mode="nearest")
    ) & (spec_db > params.peak_floor_db)
    frames, bins = np.nonzero(is_peak)  # row-major: sorted by frame
    local_mean = uniform_filter(
        spec_db,
        size=(
            params.frames(params.prominence_window_s),
            params.bins(params.prominence_window_hz),
        ),
        mode="nearest",
    )
    prominence = (spec_db - local_mean)[frames, bins]
    keep = _dense_enough(
        frames,
        prominence,
        radius=params.frames(params.peak_density_window_s) // 2,
        allowed=max(1, round(params.peaks_per_second * params.peak_density_window_s)),
    )
    return frames[keep].astype(np.int64), bins[keep].astype(np.int64)


def _dense_enough(
    frames: np.ndarray, strength: np.ndarray, *, radius: int, allowed: int
) -> np.ndarray:
    """Mask of the peaks with fewer than ``allowed`` stronger peaks within
    ``radius`` frames (``frames`` sorted). Vectorised over index distance."""
    n = len(frames)
    stronger = np.zeros(n, np.int64)
    for d in range(1, n):
        a, b = frames[:-d], frames[d:]
        near = b - a <= radius
        if not near.any():  # frames are sorted: farther indices are farther apart
            break
        sa, sb = strength[:-d], strength[d:]
        stronger[:-d] += near & (sb > sa)
        stronger[d:] += near & (sa > sb)
    return stronger < allowed


def _pair_hashes(
    frames: np.ndarray, bins: np.ndarray, params: LandmarkParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(hashes, anchor frames)``: each peak paired with its first ``fan_out``
    successors in the target zone, hashed as ``(f1, f2 - f1, dt)`` bit fields."""
    max_dt = params.frames(params.target_zone_s)
    max_df = params.bins(params.target_zone_hz)
    dt_bits = max_dt.bit_length()
    df_bits = (2 * max_df).bit_length()
    hashes, anchors = [], []
    ends = np.searchsorted(frames, frames + max_dt, side="right")
    for i in range(len(frames)):
        j = np.arange(i + 1, ends[i])
        dt = frames[j] - frames[i]
        df = bins[j] - bins[i]
        ok = (dt >= _MIN_PAIR_DT_FRAMES) & (np.abs(df) <= max_df)
        j, dt, df = (
            j[ok][: params.fan_out],
            dt[ok][: params.fan_out],
            df[ok][: params.fan_out],
        )
        hashes.append(
            (bins[i] << (df_bits + dt_bits)) | ((df + max_df) << dt_bits) | dt
        )
        anchors.append(np.full(len(j), frames[i]))
    if not hashes:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(hashes).astype(np.int64), np.concatenate(anchors).astype(
        np.int64
    )


def landmarks(
    audio: AudioInput,
    *,
    sr: Optional[int] = None,
    params: LandmarkParams = DEFAULT_LANDMARK_PARAMS,
) -> Landmarks:
    """The landmark hashes of ``audio``: a reusable index for :func:`contains_song`.

    Args:
        audio: a path, a :class:`~mixing.audio.Audio`, a pydub ``AudioSegment``, or
            a float array ``(n,)`` / ``(n, channels)`` in [-1, 1] with ``sr``.
        sr: the array's sample rate (arrays only).
        params: the hashing recipe (:class:`LandmarkParams`; defaults are the
            ``DEFAULT_*`` module constants). Audio is downmixed and resampled to
            ``params.sample_rate`` first.
    """
    y = _mono_at(audio, sr=sr, target=params.sample_rate)
    frames, bins = _peaks(_spectrogram_db(y, params), params)
    hashes, anchors = _pair_hashes(frames, bins, params)
    return Landmarks(hashes, anchors, params, len(y) / params.sample_rate)


# --- matching ----------------------------------------------------------------


def _offset_histogram(clip: Landmarks, song: Landmarks) -> tuple[np.ndarray, int]:
    """``(counts per offset frame, frame of counts[0])`` over every shared hash."""
    song_hashes, song_frames = song._sorted
    lo = np.searchsorted(song_hashes, clip.hashes, side="left")
    hi = np.searchsorted(song_hashes, clip.hashes, side="right")
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.zeros(0, np.int64), 0
    # expand each clip hash into its run of song matches
    run_start = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(total)
    offsets = song_frames[run_start] - np.repeat(clip.frames, n)
    first = int(offsets.min())
    return np.bincount(offsets - first), first


def _measure(
    clip: Landmarks,
    song: Landmarks,
    *,
    offset_tolerance_s: float,
    rival_exclusion_s: float,
) -> dict:
    """K, K2, shared and the best offset (s) of ``clip`` against ``song``."""
    counts, first = _offset_histogram(clip, song)
    shared = int(counts.sum())
    if shared == 0:
        return dict(votes=0, rival_votes=0, shared=0, offset_s=None)
    p = song.params
    half = round(offset_tolerance_s / p.hop_s)
    padded = np.pad(counts, half)
    window = np.convolve(padded, np.ones(2 * half + 1, np.int64), mode="valid")
    best = int(np.argmax(window))
    rivals = np.abs(np.arange(len(window)) - best) > p.frames(rival_exclusion_s)
    rival_votes = int(window[rivals].max()) if rivals.any() else 0
    lo, hi = max(0, best - half), min(len(counts), best + half + 1)
    span = np.arange(lo, hi)
    centroid = float((span * counts[lo:hi]).sum() / max(counts[lo:hi].sum(), 1))
    return dict(
        votes=int(window[best]),
        rival_votes=rival_votes,
        shared=shared,
        offset_s=(centroid + first) * p.hop_s,
    )


def song_verdict(
    *,
    votes: int,
    rho: float,
    ratio: float,
    query_hashes: int,
    min_votes: int = DEFAULT_MIN_VOTES,
    min_rho: float = DEFAULT_MIN_RHO,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_absent_votes: int = DEFAULT_MAX_ABSENT_VOTES,
    min_query_hashes: int = DEFAULT_MIN_QUERY_HASHES,
) -> tuple[SongVerdict, str]:
    """``(verdict, reason)`` from measured K, rho, R -- the ONE place it is decided.

    Pure, so stored measurements can be re-judged under calibrated thresholds:

    - ``"contains"`` iff ``votes >= min_votes and rho >= min_rho and ratio >= min_ratio``;
    - ``"absent"`` iff ``votes <= max_absent_votes`` and the clip offered at least
      ``min_query_hashes`` hashes (too few landmarks is no evidence of absence);
    - ``"undecided"`` otherwise.

    The defaults are the research's starting values, **not calibrated**.
    """
    if max_absent_votes >= min_votes:
        raise ValueError(
            f"max_absent_votes ({max_absent_votes}) must be below min_votes ({min_votes})"
        )
    stats = f"K={votes}, rho={rho:.3f}, R={ratio:.2f}"
    if votes >= min_votes and rho >= min_rho and ratio >= min_ratio:
        return CONTAINS, f"{stats} clear K>={min_votes}, rho>={min_rho}, R>={min_ratio}"
    if votes <= max_absent_votes and query_hashes >= min_query_hashes:
        return ABSENT, f"{stats}: K<={max_absent_votes} on {query_hashes} clip hashes"
    if votes <= max_absent_votes:
        return UNDECIDED, (
            f"{stats}, but only {query_hashes} clip hashes (< {min_query_hashes}): "
            "too little to show the song either way"
        )
    failed = [
        name
        for name, ok in (
            (f"K>={min_votes}", votes >= min_votes),
            (f"rho>={min_rho}", rho >= min_rho),
            (f"R>={min_ratio}", ratio >= min_ratio),
        )
        if not ok
    ]
    return UNDECIDED, f"{stats}: between the bars (fails {', '.join(failed)})"


def contains_song(
    clip: AudioInput,
    song: Union[AudioInput, Landmarks],
    *,
    sr: Optional[int] = None,
    song_sr: Optional[int] = None,
    params: LandmarkParams = DEFAULT_LANDMARK_PARAMS,
    query_shifts: int = DEFAULT_QUERY_SHIFTS,
    offset_tolerance_s: float = DEFAULT_OFFSET_TOLERANCE_S,
    rival_exclusion_s: float = DEFAULT_RIVAL_EXCLUSION_S,
    min_votes: int = DEFAULT_MIN_VOTES,
    min_rho: float = DEFAULT_MIN_RHO,
    min_ratio: float = DEFAULT_MIN_RATIO,
    max_absent_votes: int = DEFAULT_MAX_ABSENT_VOTES,
    min_query_hashes: int = DEFAULT_MIN_QUERY_HASHES,
) -> SongMatch:
    """Does ``clip`` contain ``song``, and where? A landmark-fingerprint verdict.

    The clip is hashed at ``query_shifts`` sub-hop phases (audfprint's
    ``--shifts``), each scored as its own query against the song; the phase with
    the most votes is reported. A clip rarely starts on the song's hop grid, and
    off-grid a peak's frame jitters by one, which changes the ``dt`` of its
    hashes; one of the phases lands close to the grid. Best-of-phases (rather
    than pooling them) keeps K the count of ONE query's consistent hashes.

    Args:
        clip: the recording to test: a path, an ``Audio``, an ``AudioSegment``,
            or an array with ``sr``.
        song: the song, the same way -- or, better, its :class:`Landmarks` from
            :func:`landmarks`, computed once for every clip.
        sr: ``clip``'s sample rate when it is an array.
        song_sr: ``song``'s sample rate when it is an array (defaults to ``sr``).
        params: hashing recipe when ``song`` is audio; a song given as
            :class:`Landmarks` hashes the clip with its own params instead.
        query_shifts: sub-hop phases the clip is tried at (1 = no shifting).
        offset_tolerance_s: hashes this close to the best offset agree with it.
        rival_exclusion_s: the rival offset is the best one farther than this.
        min_votes, min_rho, min_ratio, max_absent_votes, min_query_hashes: the
            verdict's thresholds (see :func:`song_verdict`); **uncalibrated**
            starting values from the research.

    Returns:
        A :class:`SongMatch`; ``offset_s`` is the song time at the clip's t=0.
    """
    if query_shifts < 1:
        raise ValueError(f"query_shifts must be at least 1, got {query_shifts}")
    if not isinstance(song, Landmarks):
        song = landmarks(song, sr=song_sr if song_sr is not None else sr, params=params)
    p = song.params
    y = _mono_at(clip, sr=sr, target=p.sample_rate)

    def phase(k: int) -> dict:
        lead = round(k * p.hop_length / query_shifts)
        query = landmarks(y[lead:], sr=p.sample_rate, params=p)
        m = _measure(
            query,
            song,
            offset_tolerance_s=offset_tolerance_s,
            rival_exclusion_s=rival_exclusion_s,
        )
        if m["offset_s"] is not None:  # the phase's t=0 is the clip's t=lead
            m["offset_s"] -= lead / p.sample_rate
        return dict(m, query_hashes=len(query))

    m = max((phase(k) for k in range(query_shifts)), key=lambda m: m["votes"])
    rho = m["votes"] / m["shared"] if m["shared"] else 0.0
    ratio = m["votes"] / max(m["rival_votes"], 1)
    verdict, reason = song_verdict(
        votes=m["votes"],
        rho=rho,
        ratio=ratio,
        query_hashes=m["query_hashes"],
        min_votes=min_votes,
        min_rho=min_rho,
        min_ratio=min_ratio,
        max_absent_votes=max_absent_votes,
        min_query_hashes=min_query_hashes,
    )
    return SongMatch(
        verdict=verdict,
        votes=m["votes"],
        rival_votes=m["rival_votes"],
        shared=m["shared"],
        rho=rho,
        ratio=ratio,
        offset_s=m["offset_s"],
        query_hashes=m["query_hashes"],
        reason=reason,
    )
