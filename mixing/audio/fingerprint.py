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
- ``ratio`` (**R**): ``K / K2``, ``K2`` the best rival offset more than
  ``rival_exclusion_s`` away (a rival of zero counts as one, so R is finite). A
  looped bar shares the clip's hashes with every repetition: K is high, R is ~1;
- ``coverage``: ``K`` over the clip hashes that, at that offset, overlap the song;
- ``rho``: ``K / shared``. Reported, not judged: audfprint's "chance makes < 1 %
  of common hashes consistent" [2] holds when a query is searched against a large
  database, where chance hashes are many. Against one song the shared hashes are
  few, and rho is often 0.3-1.0 for an unrelated clip too.

and calls one of THREE verdicts (:func:`song_verdict`, :class:`SongThresholds`):

- ``"contains"``: K >= 12 and R >= 5;
- ``"absent"``: K <= 5 on a clip with at least 240 hashes (enough to have shown
  the song), and no speed probe (:class:`MatchParams`, +-1-3 %) finds it;
- ``"undecided"`` otherwise -- a third outcome, never a pass. :class:`SongMatch`
  has no truth value, so "undecided" cannot slip through an ``if``.

**Measured, on synthetic media only** (the tests' generator and the review's):

- 115 hard negatives -- 90 unrelated songs sharing one 5- or 7-note scale and one
  0.25/0.5 s beat grid, 20 unrelated songs over a 3-octave range, 5 noise clips:
  **0 "contains"**, 63 "absent", 52 "undecided". These negatives reach K = 26
  and R = 4.67, so the research's starting R >= 2 called 7 of them "contains".
  That is why the bar is 5.
- 98 true cuts (noise to 0 dB SNR, -20 dB gain, a 1 kHz low-pass, all three at
  once, on off-grid starts, the grid songs included, 60 s clips, a song inside
  a longer recording): 87 "contains" at the right offset, 11 "undecided" (R
  1.9-4.9), none "absent". Coverage went as low as 0.010 on these and as high
  as 0.067 on the negatives, so it does not separate them and is not judged.
- Speed: +-0.5 % (resampled) is "contains" (32 of 32). At +-2-3 %, and at a
  +0.5 semitone transposition, a probe finds the song and the verdict is
  "undecided". Outside the probes (a transposition of a semitone or more) and
  under heavy clipping (x100), a real copy can read **"absent"**: "absent" means
  "not this recording within +-3 % of its speed", not "not this song".

**The thresholds are uncalibrated.** Wang's procedure [1] is to collect the score
distribution on *known non-matching* clips (bar-periodic aliases and same-key
songs included), and to pick the threshold that meets a target false-positive
rate. Do that on real recordings before letting this vote, and do it with
``query_shifts`` as shipped: the best of 4 phases is 4 tries at the null. Every
threshold is a field of :class:`SongThresholds`, and :func:`song_verdict`
re-judges a stored :meth:`SongMatch.to_dict` without rehashing.

Two choices beyond the textbook recipe, both measured on synthetic cuts (see
:func:`_peaks` and :func:`contains_song`): the constellation's density cap is
*local*, so a clip keeps the peaks its song had at the same place (a global or
block cap lost most of them under a low-pass); and the clip is tried at
``query_shifts`` sub-hop phases, with the best one reported.

Hash a song once and reuse it: :func:`landmarks` returns a :class:`Landmarks`
index, which :func:`contains_song` accepts in place of the song's audio. The clip
is then hashed with the index's own :class:`LandmarkParams`, so the two always agree.

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

from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Union

import numpy as np

from ._pcm import resample, segment_samples
from .audio_util import AudioSource

if TYPE_CHECKING:
    from .audio_ops import Audio

__all__ = [
    "Landmarks",
    "LandmarkParams",
    "SongMatch",
    "SongVerdict",
    "SongThresholds",
    "MatchParams",
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
#: Speed changes (fractions) a clip about to be called "absent" is re-tried at,
#: undone by resampling: a copy 1-3 % off speed (a camera's clock, a sped-up
#: upload) loses every hash at its true speed, and reads as absent without this.
#: Finding the song at one of them makes the verdict "undecided", never "contains".
DEFAULT_SPEED_PROBES = (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03)
#: Resampling ratios are expressed over this denominator (0.1 % resolution).
_SPEED_PROBE_RESOLUTION = 1000

#: Hashes within this of the best offset agree with it (s): +- one hop, the jitter
#: of a peak's frame when the clip does not start on the song's hop grid.
DEFAULT_OFFSET_TOLERANCE_S = 0.025
#: A rival offset is the best one more than this from the winner (s).
DEFAULT_RIVAL_EXCLUSION_S = 0.5
#: K at or above which the song may be called present (research: start at 12).
DEFAULT_MIN_VOTES = 12
#: R at or above which the winning offset clearly beats every rival. The research
#: started at 2; synthetic hard negatives (unrelated songs in one key on one beat
#: grid) reached R = 4.67 with K >= 12, so 2 called 7 of 115 of them "contains".
DEFAULT_MIN_RATIO = 5.0
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
class MatchParams:
    """How a clip's hashes are matched against a song's (defaults: ``DEFAULT_*``).

    Attributes:
        query_shifts: sub-hop phases the clip is hashed at, best one reported.
        offset_tolerance_s: hashes this close to the best offset agree with it.
        rival_exclusion_s: the rival offset is the best one farther than this.
        speed_probes: speed changes re-tried before calling "absent" (``()`` = none).
    """

    query_shifts: int = DEFAULT_QUERY_SHIFTS
    offset_tolerance_s: float = DEFAULT_OFFSET_TOLERANCE_S
    rival_exclusion_s: float = DEFAULT_RIVAL_EXCLUSION_S
    speed_probes: tuple[float, ...] = DEFAULT_SPEED_PROBES

    def __post_init__(self):
        if self.query_shifts < 1:
            raise ValueError(
                f"query_shifts must be at least 1, got {self.query_shifts}"
            )


DEFAULT_MATCH_PARAMS = MatchParams()


@dataclass(frozen=True)
class SongThresholds:
    """The bars :func:`song_verdict` applies (defaults: ``DEFAULT_*``). **Uncalibrated.**

    Attributes:
        min_votes: K at or above which "contains" is possible.
        min_ratio: R at or above which "contains" is possible.
        max_absent_votes: K at or below which "absent" is possible.
        min_query_hashes: clip hashes needed before "absent" may be called.
    """

    min_votes: int = DEFAULT_MIN_VOTES
    min_ratio: float = DEFAULT_MIN_RATIO
    max_absent_votes: int = DEFAULT_MAX_ABSENT_VOTES
    min_query_hashes: int = DEFAULT_MIN_QUERY_HASHES

    def __post_init__(self):
        if self.max_absent_votes >= self.min_votes:
            raise ValueError(
                f"max_absent_votes ({self.max_absent_votes}) must be below "
                f"min_votes ({self.min_votes})"
            )


DEFAULT_SONG_THRESHOLDS = SongThresholds()


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
        overlap_hashes: the clip hashes that, at ``offset_s``, fall inside the song.
        coverage: ``votes / overlap_hashes``: how much of the clip, where it
            overlaps the song, the song accounts for (0 when nothing overlaps).
        rho: ``votes / shared`` (reported, not judged: in a one-to-one match the
            shared hashes are few, so rho is high for noise too).
        ratio: R, ``votes / max(rival_votes, 1)``.
        offset_s: song time at the clip's t=0 (``t_song - t_clip``) for the best
            offset; ``None`` when nothing is shared. Only meaningful with
            ``"contains"``.
        query_hashes: the clip's own hash count, at the reported phase (the
            evidence it could offer).
        off_speed: the speed change (fraction) at which a speed probe found the
            song, turning "absent" into "undecided"; ``None`` otherwise.
        probed_speeds: the speed changes actually tried (empty unless the
            verdict would otherwise have been "absent").
        reason: one line on why this verdict.
        landmark_params, match_params, thresholds: what produced it.
    """

    verdict: SongVerdict
    votes: int
    rival_votes: int
    shared: int
    overlap_hashes: int
    coverage: float
    rho: float
    ratio: float
    offset_s: Optional[float]
    query_hashes: int
    off_speed: Optional[float] = None
    probed_speeds: tuple[float, ...] = ()
    reason: str = field(default="", compare=False)
    landmark_params: LandmarkParams = DEFAULT_LANDMARK_PARAMS
    match_params: MatchParams = DEFAULT_MATCH_PARAMS
    thresholds: SongThresholds = DEFAULT_SONG_THRESHOLDS

    def __bool__(self) -> bool:
        raise TypeError(
            "SongMatch has no truth value: 'undecided' is not a pass. "
            f"Compare match.verdict == {CONTAINS!r} (verdict is {self.verdict!r})."
        )

    def to_dict(self) -> dict:
        """Plain-JSON form (for a record that stores the measurement)."""
        return asdict(self)


# --- input -------------------------------------------------------------------

#: What a clip or a song may be: a path, a pydub ``AudioSegment``, an array with
#: ``sr=`` (float in [-1, 1], or integer PCM, scaled by its full scale), or a
#: :class:`~mixing.audio.Audio` (incl. a slice of one).
AudioInput = Union[AudioSource, "Audio"]


def _as_float(data: np.ndarray) -> np.ndarray:
    """Samples as float64 in [-1, 1]: integer PCM is divided by its full scale
    (unsigned PCM is offset-binary, so it is centred first); floats pass through."""
    if np.issubdtype(data.dtype, np.integer):
        bits = np.iinfo(data.dtype).bits
        full_scale = float(1 << (bits - 1))
        centred = data.astype(np.float64)
        if np.issubdtype(data.dtype, np.unsignedinteger):
            centred -= full_scale
        return centred / full_scale
    return np.asarray(data, dtype=np.float64)


def _mono_at(audio: AudioInput, *, sr: Optional[int], target: int) -> np.ndarray:
    """``audio`` as mono float samples in [-1, 1] at ``target`` Hz (anti-aliased)."""
    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError("an array needs its sample rate: pass sr=")
        data = _as_float(audio)
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


def _measure(clip: Landmarks, song: Landmarks, match: MatchParams) -> dict:
    """K, K2, shared, overlap and the best offset of ``clip`` against ``song``."""
    counts, first = _offset_histogram(clip, song)
    shared = int(counts.sum())
    if shared == 0:
        return dict(votes=0, rival_votes=0, shared=0, overlap_hashes=0, offset_s=None)
    p = song.params
    half = round(match.offset_tolerance_s / p.hop_s)
    padded = np.pad(counts, half)
    window = np.convolve(padded, np.ones(2 * half + 1, np.int64), mode="valid")
    best = int(np.argmax(window))
    rivals = np.abs(np.arange(len(window)) - best) > p.frames(match.rival_exclusion_s)
    rival_votes = int(window[rivals].max()) if rivals.any() else 0
    lo, hi = max(0, best - half), min(len(counts), best + half + 1)
    span = np.arange(lo, hi)
    centroid = float((span * counts[lo:hi]).sum() / max(counts[lo:hi].sum(), 1))
    # the clip hashes that, placed at this offset, fall inside the song
    in_song = clip.frames + (best + first)
    overlap = int(((in_song >= 0) & (in_song * p.hop_s <= song.duration_s)).sum())
    return dict(
        votes=int(window[best]),
        rival_votes=rival_votes,
        shared=shared,
        overlap_hashes=overlap,
        offset_s=(centroid + first) * p.hop_s,
    )


def _found_off_speed(
    y: np.ndarray,
    song: Landmarks,
    *,
    match: MatchParams,
    thresholds: SongThresholds,
) -> tuple[Optional[float], tuple[float, ...]]:
    """``(off_speed, probed)``: the first of ``match.speed_probes`` at which ``y``,
    its speed change undone, clears the "contains" bars on K and R (one phase
    each: a probe looks for a cluster, it does not place the clip), or ``None``;
    and the speed changes tried up to it."""
    p = song.params
    probed: list[float] = []
    for change in match.speed_probes:
        probed.append(change)
        stretched = resample(
            y, _SPEED_PROBE_RESOLUTION, round(_SPEED_PROBE_RESOLUTION * (1 + change))
        )
        m = _measure(landmarks(stretched, sr=p.sample_rate, params=p), song, match)
        ratio = m["votes"] / max(m["rival_votes"], 1)
        if m["votes"] >= thresholds.min_votes and ratio >= thresholds.min_ratio:
            return change, tuple(probed)
    return None, tuple(probed)


def _field(record: Mapping, name: str, default):
    """``record[name]`` as a plain mapping (a dataclass, a dict, or absent)."""
    value = record.get(name)
    if value is None:
        return default
    return asdict(value) if not isinstance(value, Mapping) else value


def _stored_thresholds(record: Mapping) -> SongThresholds:
    """The :class:`SongThresholds` a record was judged under (default if none)."""
    stored = _field(record, "thresholds", {})
    known = {f.name for f in fields(SongThresholds)}
    return SongThresholds(**{k: v for k, v in stored.items() if k in known})


def _base_verdict(record: Mapping, t: SongThresholds) -> tuple[SongVerdict, str]:
    """The verdict from K, R and the clip's hash count alone (no speed probes)."""
    votes, ratio, query_hashes = (
        record["votes"],
        record["ratio"],
        record["query_hashes"],
    )
    stats = f"K={votes}, R={ratio:.2f}, coverage={record.get('coverage', 0.0):.3f}"
    passes = {
        f"K>={t.min_votes}": votes >= t.min_votes,
        f"R>={t.min_ratio}": ratio >= t.min_ratio,
    }
    if all(passes.values()):
        return CONTAINS, f"{stats}: clears {', '.join(passes)}"
    if votes <= t.max_absent_votes and query_hashes >= t.min_query_hashes:
        return ABSENT, f"{stats}: K<={t.max_absent_votes} on {query_hashes} clip hashes"
    if votes <= t.max_absent_votes:
        return UNDECIDED, (
            f"{stats}, but only {query_hashes} clip hashes (< {t.min_query_hashes}): "
            "too little to show the song either way"
        )
    failed = ", ".join(name for name, ok in passes.items() if not ok)
    return UNDECIDED, f"{stats}: between the bars (fails {failed})"


def song_verdict(
    measurement: Union[SongMatch, Mapping],
    *,
    thresholds: Optional[SongThresholds] = None,
) -> tuple[SongVerdict, str]:
    """``(verdict, reason)`` from a measurement -- the ONE place it is decided.

    ``measurement`` is a :class:`SongMatch` or its :meth:`~SongMatch.to_dict`
    (JSON round-tripped or not), so a stored record is re-judged in one call,
    without rehashing:

    - ``"contains"`` iff ``votes >= min_votes`` and ``ratio >= min_ratio``;
    - ``"absent"`` iff ``votes <= max_absent_votes``, the clip offered at least
      ``min_query_hashes`` hashes (too few landmarks is no evidence of absence),
      and the speed probes ran and found nothing: a recorded ``off_speed`` makes it
      "undecided", and so does a record whose configured probes
      (``match_params.speed_probes``) never ran -- re-judging under other
      thresholds can reach "absent" where the live call did not probe;
    - ``"undecided"`` otherwise.

    ``thresholds`` defaults to the ones stored IN the record (or
    :class:`SongThresholds` for a record without them); pass them to override.
    The defaults are **not calibrated**.
    """
    record = (
        measurement.to_dict() if isinstance(measurement, SongMatch) else measurement
    )
    t = thresholds if thresholds is not None else _stored_thresholds(record)
    verdict, reason = _base_verdict(record, t)
    if verdict != ABSENT:
        return verdict, reason
    off_speed = record.get("off_speed")
    if off_speed is not None:
        return UNDECIDED, (
            f"{reason}, but the song shows at {off_speed:+.1%} speed: an "
            "off-speed copy is not evidence of absence"
        )
    configured = tuple(_field(record, "match_params", {}).get("speed_probes", ()))
    probed = tuple(record.get("probed_speeds", ()))
    if not set(configured) <= set(probed):
        return UNDECIDED, (
            f"{reason}, but the speed probes {configured} did not run for this "
            "record: absence at the song's speed only"
        )
    return verdict, reason


def contains_song(
    clip: AudioInput,
    song: Union[AudioInput, Landmarks],
    *,
    sr: Optional[int] = None,
    song_sr: Optional[int] = None,
    params: Optional[LandmarkParams] = None,
    match: MatchParams = DEFAULT_MATCH_PARAMS,
    thresholds: SongThresholds = DEFAULT_SONG_THRESHOLDS,
) -> SongMatch:
    """Does ``clip`` contain ``song``, and where? A landmark-fingerprint verdict.

    The clip is hashed at ``match.query_shifts`` sub-hop phases (audfprint's
    ``--shifts``), each scored as its own query against the song; the phase with
    the most votes is reported. A clip rarely starts on the song's hop grid, and
    off-grid a peak's frame jitters by one, which changes the ``dt`` of its
    hashes; one of the phases lands close to the grid. Best-of-phases (rather
    than pooling them) keeps K the count of ONE query's consistent hashes --
    but it is ``query_shifts`` tries at the null, so calibrate with the shifts
    you will ship.

    Args:
        clip: the recording to test (see :data:`AudioInput`).
        song: the song, the same way -- or, better, its :class:`Landmarks` from
            :func:`landmarks`, computed once for every clip.
        sr: ``clip``'s sample rate when it is an array.
        song_sr: ``song``'s sample rate when it is an array (defaults to ``sr``);
            an error with a :class:`Landmarks` song.
        params: hashing recipe when ``song`` is audio (default
            :data:`DEFAULT_LANDMARK_PARAMS`). With a :class:`Landmarks` song the
            clip is hashed with the index's own params; passing different ones
            is an error.
        match: how hashes are matched (:class:`MatchParams`).
        thresholds: the verdict's bars (:class:`SongThresholds`; **uncalibrated**).

    Returns:
        A :class:`SongMatch`; ``offset_s`` is the song time at the clip's t=0.
    """
    if isinstance(song, Landmarks):
        if song_sr is not None:
            raise ValueError("song_sr= is for a song given as an array, not Landmarks")
        if params is not None and params != song.params:
            raise ValueError(
                "params differ from the Landmarks index's own; the clip must be "
                "hashed like the song (omit params=, or re-index the song)"
            )
    else:
        song = landmarks(
            song,
            sr=song_sr if song_sr is not None else sr,
            params=params if params is not None else DEFAULT_LANDMARK_PARAMS,
        )
    p = song.params
    y = _mono_at(clip, sr=sr, target=p.sample_rate)

    def phase(k: int) -> dict:
        lead = round(k * p.hop_length / match.query_shifts)
        query = landmarks(y[lead:], sr=p.sample_rate, params=p)
        m = _measure(query, song, match)
        if m["offset_s"] is not None:  # the phase's t=0 is the clip's t=lead
            m["offset_s"] -= lead / p.sample_rate
        return dict(m, query_hashes=len(query))

    m = max((phase(k) for k in range(match.query_shifts)), key=lambda m: m["votes"])
    measured = dict(
        votes=m["votes"],
        rival_votes=m["rival_votes"],
        shared=m["shared"],
        overlap_hashes=m["overlap_hashes"],
        coverage=m["votes"] / m["overlap_hashes"] if m["overlap_hashes"] else 0.0,
        rho=m["votes"] / m["shared"] if m["shared"] else 0.0,
        ratio=m["votes"] / max(m["rival_votes"], 1),
        offset_s=m["offset_s"],
        query_hashes=m["query_hashes"],
    )
    measured.update(
        off_speed=None, probed_speeds=(), match_params=match, thresholds=thresholds
    )
    if _base_verdict(measured, thresholds)[0] == ABSENT:  # probe before calling it
        off_speed, probed = _found_off_speed(
            y, song, match=match, thresholds=thresholds
        )
        measured.update(off_speed=off_speed, probed_speeds=probed)
    verdict, reason = song_verdict(measured)
    return SongMatch(verdict=verdict, reason=reason, landmark_params=p, **measured)
