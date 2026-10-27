"""Voice diagnosis (f0, loudness, timbre per speaker) and pitch shifting.

Two pieces, in the order they are needed when "one speaker sounds too low":

1. :func:`voice_profile` measures, per label (speaker), the **median f0** (librosa
   ``pyin``, voiced frames only), the **integrated loudness** (BS.1770, see
   :func:`mixing.audio.leveling.loudness`), the **spectral centroid** and the
   **low-band ratio** (energy of the 60-150 Hz "boom" band relative to the full
   band, in dB). That diagnosis decides whether pitch is the problem at all: a
   voice that *sounds* low is often just quieter and boomier (fix: EQ plus
   :func:`mixing.audio.level_turns`), not lower in f0.
2. :func:`shift_pitch` shifts pitch by a number of semitones, keeping duration
   exactly and (by default) the integrated loudness. ``method="rubberband"``
   uses ffmpeg's ``rubberband`` filter, which can **preserve formants** (the
   voice stays the same person, just higher/lower); it needs an ffmpeg built
   with ``librubberband``. ``method="resample"`` needs only stock ffmpeg
   (``asetrate`` + ``aresample`` + ``atempo``) and ``method="librosa"`` (phase
   vocoder + resampling, extra ``beats``); both move formants with the pitch.
   ``method="auto"`` prefers rubberband and falls back to resample, with a
   warning when formants were asked to be preserved. The ffmpeg methods'
   delay depends on the shift ratio; it is measured per ratio
   (:func:`shift_latency`) and removed.

Spans are given in **seconds** whatever the ``Audio.time_unit``; all slicing is
done in sample frames of the underlying segment (as in
:mod:`mixing.audio.turns`). Processing runs on float samples and is converted
back to the input's integer sample width only at the end.

>>> from pydub import AudioSegment
>>> profile = voice_profile(Audio(AudioSegment.silent(1000)), [("a", 0.0, 0.5)],
...                         f0=False)
>>> round(profile["a"]["duration_s"], 2), profile["a"]["loudness_lufs"]
(0.5, None)
"""

from __future__ import annotations

import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Hashable, Mapping, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe, require_package
from .audio_ops import Audio
from .leveling import integrated_lufs
from .samples import as_audio, ffmpeg_float, resample, segment_samples, to_audio
from .turns import TURN_BOUND_TOLERANCE_S

Label = Hashable
Span = tuple[float, float]  # (start_s, end_s)
#: ``[(label, start_s, end_s), ...]`` (the ``turns`` shape of ``level_turns``) or
#: ``{label: [(start_s, end_s), ...]}``.
Spans = Union[Sequence[tuple[Label, float, float]], Mapping[Label, Sequence[Span]]]
#: ``strategy(samples (n, channels) float, rate, semitones, preserve_formants) -> samples``
PitchStrategy = Callable[[np.ndarray, int, float, bool], np.ndarray]
#: ``tracker(mono samples, rate) -> (f0_hz per frame, voiced flag per frame)``;
#: ``f0_hz`` is only read where the flag is true, so a tracker applies its own
#: voicing threshold. A tracker may carry a ``min_samples`` attribute, the
#: shortest input it can track; shorter spans are skipped (default
#: ``MIN_F0_SPAN_S``). Any pitch tracker fits behind this, e.g. ``denote``::
#:
#:     def denote_tracker(*, threshold=0.5, **kwargs):
#:         def track(y, fs):
#:             pitch = denote.get_pitch(y, sr=fs, **kwargs)
#:             return pitch.frequencies, pitch.confidence >= threshold
#:         return track
F0Tracker = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


# --- voice_profile defaults ---------------------------------------------------
#: pyin search range (Hz): covers low male (~65 Hz) to high female/child speech.
DEFAULT_F0_FMIN_HZ = 65.0
DEFAULT_F0_FMAX_HZ = 600.0
#: Rate (Hz) the f0 tracker runs at; ample for f0 <= 600 Hz and keeps pyin fast.
DEFAULT_F0_SAMPLE_RATE = 16000
#: pyin frame (samples at ``DEFAULT_F0_SAMPLE_RATE``): 128 ms, > 8 periods of fmin.
DEFAULT_F0_FRAME_LENGTH = 2048
#: Shortest span (s) handed to a tracker without a ``min_samples`` attribute:
#: one default pyin frame.
MIN_F0_SPAN_S = DEFAULT_F0_FRAME_LENGTH / DEFAULT_F0_SAMPLE_RATE
#: librosa ``pyin``'s default hop is a quarter frame.
PYIN_HOPS_PER_FRAME = 4
#: pyin frames flagged voiced but less likely than this are dropped: on noise
#: pyin flags some frames (near ``fmin``) voiced at ~0.01 probability, while
#: real voicing scores well above 0.2.
DEFAULT_MIN_VOICED_PROB = 0.1
#: The "boom" band whose energy share is ``low_band_db``.
DEFAULT_LOW_BAND_HZ = (60.0, 150.0)
#: Welch segment length (samples) for centroid / low-band; ~21 Hz bins at 44.1 kHz.
DEFAULT_WELCH_NPERSEG = 2048
#: Label used when ``voice_profile`` is called with ``spans=None``.
WHOLE_AUDIO_LABEL = "all"

# --- shift_pitch defaults -----------------------------------------------------
#: Accepted ``|semitones|`` (two octaves).
MAX_SEMITONES = 24.0
SEMITONES_PER_OCTAVE = 12.0
PITCH_METHODS = ("auto", "rubberband", "resample", "librosa")
#: Largest processing delay (s), either sign, searched for and removed.
MAX_SHIFT_LATENCY_S = 0.5
#: Silence (s) appended to the input of an ffmpeg shifter so its last block is
#: flushed (no tail is lost), on top of the measured delay.
SHIFT_FLUSH_S = 0.25
#: A custom strategy's output may differ from the input length by at most this
#: much (s); within it the result is zero-padded or cut, beyond it ``ValueError``.
LENGTH_TOLERANCE_S = 0.005
#: rubberband ``pitchq`` values. Measured on a synthetic vowel (formants at 700 and
#: 1800 Hz, f0 120 Hz) at +3 / -3 semitones, all three move the spectral centroid
#: alike (1106 -> ~1023 / 773-858 Hz): the darkening comes from preserving formants,
#: not from ``pitchq``, so the faster ``"speed"`` stays the default.
RUBBERBAND_PITCHQ = ("speed", "quality", "consistency")
DEFAULT_RUBBERBAND_PITCHQ = "speed"
#: ``atempo`` accepts a factor in this range per stage (older ffmpeg builds).
ATEMPO_RANGE = (0.5, 2.0)
# The delay of a pitch shifter depends on the ratio (rubberband: ~1024/ratio -
# 1024 samples plus a signal-dependent part), so it is measured per filter on a
# seeded voice-like probe: harmonic bursts (random f0, length, gap). A pitch
# shift leaves the amplitude envelope in place, so the delay is the lag that
# best aligns the smoothed energy envelopes of probe and output.
LATENCY_PROBE_S = 8.0
#: The delay is the median of the lags measured on one probe per seed, so no
#: single probe sets it: per-probe lags of one filter spread by ~5 ms (rubberband
#: at -12 and -24 st, 44.1 kHz), and three seeds still left the median ~5 ms off.
LATENCY_PROBE_SEEDS = (1770, 1771, 1772, 1773, 1774)
LATENCY_PROBE_F0_HZ = (80.0, 300.0)
LATENCY_PROBE_BURST_S = (0.06, 0.3)
LATENCY_PROBE_GAP_S = (0.05, 0.25)
LATENCY_PROBE_HARMONICS = 8
LATENCY_PROBE_AMPLITUDE = 0.2
#: Moving-average length (s) of the energy envelope used for the delay.
LATENCY_ENVELOPE_S = 0.005
#: Peak (linear, ~-0.1 dBFS) the result is scaled to when the shifted, loudness-
#: restored signal would exceed full scale; just under 1.0 so that rounding to
#: the integer sample width (8-bit included) cannot clip.
CLIP_CEILING = 10.0 ** (-0.1 / 20.0)
RUBBERBAND_INSTALL_HINT = (
    "ffmpeg's 'rubberband' filter is not available in {exe!r}. Install an ffmpeg "
    "built with --enable-librubberband (macOS: `brew tap homebrew-ffmpeg/ffmpeg && "
    "brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-rubberband`; Debian/Ubuntu: "
    "`apt install ffmpeg`, whose build includes it) and point mixing at it with "
    "the IMAGEIO_FFMPEG_EXE environment variable, or use method='resample' or "
    "'librosa' (neither preserves formants)."
)


# --- spans --------------------------------------------------------------------


def _spans_by_label(spans: Optional[Spans], duration: float) -> dict:
    """``{label: [(start, end), ...]}`` merged (union), sorted, range-checked."""
    if spans is None:
        grouped = {WHOLE_AUDIO_LABEL: [(0.0, duration)]}
    elif isinstance(spans, Mapping):
        grouped = {}
        for label, ranges in spans.items():
            if (
                isinstance(ranges, tuple)
                and len(ranges) == 2
                and all(isinstance(x, (int, float)) for x in ranges)
            ):
                raise TypeError(
                    f"spans[{label!r}] is a single span {ranges!r}; map each label "
                    f"to a list of (start_s, end_s) spans: {{{label!r}: [{ranges!r}]}}"
                )
            grouped[label] = list(ranges)
    else:
        grouped = {}
        for label, start, end in spans:
            grouped.setdefault(label, []).append((start, end))

    def union(label, ranges):
        merged: list[list[float]] = []
        for start, end in sorted(ranges):
            if not 0 <= start < end <= duration + TURN_BOUND_TOLERANCE_S:
                raise ValueError(
                    f"bad span {(label, start, end)} for {duration:.2f}s audio"
                )
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, min(end, duration)])
        return [tuple(r) for r in merged]

    return {label: union(label, ranges) for label, ranges in grouped.items()}


# --- measurements ---------------------------------------------------------------


def _band_stats(
    mono: np.ndarray, fs: int, *, low_band: Span, nperseg: int
) -> tuple[Optional[float], Optional[float]]:
    """``(spectral centroid Hz, low-band share dB)`` of ``mono``; ``None`` if silent."""
    from scipy.signal import welch

    if len(mono) == 0:
        return None, None
    freqs, psd = welch(mono, fs=fs, nperseg=min(nperseg, len(mono)))
    total = float(psd.sum())
    if total <= 0:
        return None, None
    centroid = float((freqs * psd).sum() / total)
    lo, hi = low_band
    band = float(psd[(freqs >= lo) & (freqs <= hi)].sum())
    low_db = 10.0 * np.log10(band / total) if band > 0 else -np.inf
    return centroid, (float(low_db) if np.isfinite(low_db) else None)


def pyin_tracker(
    *,
    fmin: float = DEFAULT_F0_FMIN_HZ,
    fmax: float = DEFAULT_F0_FMAX_HZ,
    frame_length: int = DEFAULT_F0_FRAME_LENGTH,
    min_voiced_prob: float = DEFAULT_MIN_VOICED_PROB,
) -> F0Tracker:
    """The default :data:`F0Tracker`: librosa ``pyin`` (extra ``beats``).

    Frames pyin flags voiced with probability below ``min_voiced_prob`` count
    as unvoiced; digital silence is all unvoiced frames without running pyin.
    Its ``min_samples`` is ``frame_length``: a shorter span is not tracked.
    """

    def track(y: np.ndarray, fs: int):
        if not np.any(y):  # pyin's frame count (centred frames, default hop)
            n = 1 + len(y) // (frame_length // PYIN_HOPS_PER_FRAME)
            return np.full(n, np.nan), np.zeros(n, dtype=bool)
        librosa = require_package("librosa")
        f0, flags, probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=fs, frame_length=frame_length
        )
        return f0, flags & (probs >= min_voiced_prob)

    track.min_samples = frame_length
    return track


def _voiced_f0(
    pieces: Sequence[np.ndarray],
    fs: int,
    *,
    tracker: F0Tracker,
    sample_rate: int,
) -> tuple[Optional[float], Optional[float]]:
    """``(median f0 over voiced frames, voiced fraction)`` of mono ``pieces``.

    Each piece (one span) is tracked separately, so span seams make no false
    pitch; pieces shorter than the tracker's ``min_samples`` (at
    ``sample_rate``; default ``MIN_F0_SPAN_S``) are skipped. ``(None, None)`` when no piece is long enough; the f0 is ``None``
    when no frame is voiced.
    """
    min_samples = getattr(tracker, "min_samples", round(MIN_F0_SPAN_S * sample_rate))
    voiced, n_frames = [], 0
    for piece in pieces:
        y = resample(piece, fs, sample_rate)
        if len(y) < min_samples:
            continue
        f0, flags = (np.ravel(a) for a in tracker(y, sample_rate))
        if len(f0) != len(flags):
            raise ValueError(
                f"f0 tracker returned {len(f0)} f0 values but {len(flags)} voiced flags"
            )
        n_frames += len(f0)
        voiced.append(f0[flags.astype(bool) & np.isfinite(f0)])
    values = np.concatenate(voiced) if voiced else np.array([])
    if not n_frames:
        return None, None
    return (float(np.median(values)) if len(values) else None), len(values) / n_frames


def _deprecated_pyin_args(**kwargs) -> dict:
    """The pyin arguments ``voice_profile`` was given (``None`` = not given),
    with a ``DeprecationWarning`` when there are any."""
    given = {k: v for k, v in kwargs.items() if v is not None}
    if given:
        warnings.warn(
            f"voice_profile({', '.join(given)}=...) is deprecated; pass "
            "f0_tracker=pyin_tracker(...) instead",
            DeprecationWarning,
            stacklevel=3,
        )
    return given


def voice_profile(
    audio: Union[str, Path, Audio],
    spans: Optional[Spans] = None,
    *,
    f0: bool = True,
    f0_tracker: Optional[F0Tracker] = None,
    f0_sample_rate: int = DEFAULT_F0_SAMPLE_RATE,
    low_band: Span = DEFAULT_LOW_BAND_HZ,
    welch_nperseg: int = DEFAULT_WELCH_NPERSEG,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    f0_frame_length: Optional[int] = None,
    min_voiced_prob: Optional[float] = None,
) -> dict:
    """Pitch, loudness and timbre of each labelled part of ``audio``.

    All spans of one label are measured together, over their **union**
    (overlapping spans of a label are merged; nothing is averaged per span).

    Args:
        audio: the recording (path or :class:`Audio`; any ``time_unit``).
        spans: ``[(label, start_s, end_s), ...]`` -- the ``turns`` of
            :func:`mixing.audio.level_turns` -- or ``{label: [(start_s, end_s),
            ...]}``, in seconds. ``None`` measures the whole audio under the
            label ``WHOLE_AUDIO_LABEL``. Spans of different labels may overlap.
        f0: estimate f0; ``False`` skips it (``f0_hz`` and ``voiced_fraction``
            are then ``None``).
        f0_tracker: the :data:`F0Tracker`; default :func:`pyin_tracker` (needs
            librosa, extra ``beats``). Configure pyin through it:
            ``f0_tracker=pyin_tracker(fmin=50, fmax=400)``. pyin's default
            range is ``DEFAULT_F0_FMIN_HZ``-``DEFAULT_F0_FMAX_HZ``; a voice
            above ``fmax`` is reported at a subharmonic (e.g. half its f0), not
            as unvoiced.
        f0_sample_rate: the rate spans are resampled to before tracking.
        low_band: ``(lo_hz, hi_hz)`` of the band reported by ``low_band_db``.
        welch_nperseg: Welch segment length for centroid and low-band ratio.
        fmin, fmax, f0_frame_length, min_voiced_prob: deprecated; pass them to
            :func:`pyin_tracker` as ``f0_tracker``.

    Returns:
        ``{label: {...}}`` of plain JSON-able dicts with keys

        - ``duration_s``: total length of the label's (merged) spans;
        - ``f0_hz``: median f0 over voiced frames (``None`` if none are voiced);
        - ``voiced_fraction``: share of pyin frames judged voiced (``None``,
          like ``f0_hz``, when no span is one pyin frame long: 128 ms by
          default);
        - ``loudness_lufs``: BS.1770 integrated loudness of the union
          (``None`` when below the -70 LUFS gate, i.e. silence);
        - ``spectral_centroid_hz``: power-weighted mean frequency;
        - ``low_band_db``: ``10 log10`` of the ``low_band`` energy over the
          full-band energy (higher = boomier; ``None`` for silence). The
          default 60-150 Hz band overlaps a low male voice's fundamental
          (~85-155 Hz), so compare it between takes of the same speaker, or
          read it next to ``f0_hz``, rather than as pure "boom".
    """
    audio = as_audio(audio)
    segment = audio._get_segment()
    fs = segment.frame_rate
    data = segment_samples(segment)
    grouped = _spans_by_label(spans, len(data) / fs)
    pyin_args = _deprecated_pyin_args(
        fmin=fmin,
        fmax=fmax,
        frame_length=f0_frame_length,
        min_voiced_prob=min_voiced_prob,
    )
    if f0_tracker is None:
        f0_tracker = pyin_tracker(**pyin_args)
    elif pyin_args:
        raise ValueError(
            "fmin, fmax, f0_frame_length and min_voiced_prob configure the default "
            "pyin tracker only; pass them to pyin_tracker() as f0_tracker"
        )

    def profile(ranges) -> dict:
        frames = [
            (int(round(s * fs)), min(len(data), int(round(e * fs)))) for s, e in ranges
        ]
        pieces = [data[a:b] for a, b in frames if b > a]
        union = np.concatenate(pieces) if pieces else np.zeros((0, data.shape[1]))
        mono = union.mean(axis=1)
        lufs = integrated_lufs(union, fs)
        centroid, low_db = _band_stats(
            mono, fs, low_band=low_band, nperseg=welch_nperseg
        )
        f0_hz = fraction = None
        if f0:
            f0_hz, fraction = _voiced_f0(
                [p.mean(axis=1) for p in pieces],
                fs,
                tracker=f0_tracker,
                sample_rate=f0_sample_rate,
            )
        return {
            "duration_s": len(union) / fs,
            "f0_hz": f0_hz,
            "voiced_fraction": fraction,
            "loudness_lufs": float(lufs) if np.isfinite(lufs) else None,
            "spectral_centroid_hz": centroid,
            "low_band_db": low_db,
        }

    return {label: profile(ranges) for label, ranges in grouped.items()}


# --- pitch shifting -------------------------------------------------------------


_FFMPEG_FILTERS_CACHE: dict = {}


def _ffmpeg_filters(exe: str) -> frozenset:
    """Names of the filters ``exe`` was built with.

    Probed once per executable; a failed probe (no such executable, or no
    filter listed) returns an empty set that is *not* cached, so a later call
    probes again.
    """
    if exe in _FFMPEG_FILTERS_CACHE:
        return _FFMPEG_FILTERS_CACHE[exe]
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-filters"], capture_output=True, text=True
        )
    except OSError:
        return frozenset()
    names = frozenset(
        parts[1]
        for parts in map(str.split, proc.stdout.splitlines())
        if len(parts) >= 3 and "->" in parts[2]
    )
    if names:
        _FFMPEG_FILTERS_CACHE[exe] = names
    return names


def has_rubberband(ffmpeg: Optional[str] = None) -> bool:
    """Whether ``ffmpeg`` (default :func:`mixing.util.ffmpeg_exe`) has ``rubberband``."""
    return "rubberband" in _ffmpeg_filters(ffmpeg or ffmpeg_exe())


def _ratio(semitones: float) -> float:
    return 2.0 ** (semitones / SEMITONES_PER_OCTAVE)


def _latency_probe(fs: int, seed: int) -> np.ndarray:
    """The seeded voice-like probe (mono float) the shift delay is measured on."""
    rng = np.random.default_rng(seed)
    n = int(round(LATENCY_PROBE_S * fs))
    t = np.arange(n) / fs
    probe = np.zeros(n)
    at = LATENCY_PROBE_BURST_S[1]
    while at < LATENCY_PROBE_S - 2 * LATENCY_PROBE_BURST_S[1]:
        length = rng.uniform(*LATENCY_PROBE_BURST_S)
        a, b = int(at * fs), int((at + length) * fs)
        f0 = rng.uniform(*LATENCY_PROBE_F0_HZ)
        wave = sum(
            np.sin(2 * np.pi * f0 * h * t[a:b] + rng.uniform(0, 2 * np.pi)) / h
            for h in range(1, LATENCY_PROBE_HARMONICS + 1)
        )
        probe[a:b] = LATENCY_PROBE_AMPLITUDE * wave / np.max(np.abs(wave))
        at += length + rng.uniform(*LATENCY_PROBE_GAP_S)
    return probe


def _energy_envelope(x: np.ndarray, fs: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    env = uniform_filter1d(x**2, max(1, int(round(LATENCY_ENVELOPE_S * fs))))
    return env - env.mean()


def _probe_lag(filters: str, fs: int, seed: int, max_lag: int) -> int:
    """Lag (samples, ``-max_lag..max_lag``) best aligning one probe's envelopes."""
    padded = np.concatenate(
        [np.zeros(max_lag), _latency_probe(fs, seed), np.zeros(max_lag)]
    )
    out = ffmpeg_float(padded[:, None], fs, filters)[:, 0]
    ref, res = _energy_envelope(padded, fs), _energy_envelope(out, fs)
    size = 1 << int(np.ceil(np.log2(len(ref) + len(res))))
    xcorr = np.fft.irfft(np.fft.rfft(res, size) * np.conj(np.fft.rfft(ref, size)), size)
    lags = np.concatenate([xcorr[-max_lag:], xcorr[: max_lag + 1]])
    return int(np.argmax(lags)) - max_lag


@lru_cache(maxsize=64)
def _shift_latency(filters: str, fs: int, exe: str) -> int:
    max_lag = int(round(MAX_SHIFT_LATENCY_S * fs))
    lag = int(
        np.median([_probe_lag(filters, fs, s, max_lag) for s in LATENCY_PROBE_SEEDS])
    )
    if abs(lag) >= max_lag:
        raise RuntimeError(
            f"could not measure the delay of ffmpeg filter {filters!r} at {fs} Hz: "
            f"the best lag sits at the search limit (MAX_SHIFT_LATENCY_S = "
            f"{MAX_SHIFT_LATENCY_S} s), so the filter outputs silence or delays "
            "by more than that"
        )
    return lag


def shift_latency(filters: str, fs: int) -> int:
    """Delay (samples, may be negative) of a pitch-shifting ffmpeg ``filters`` chain.

    The median, over the ``LATENCY_PROBE_SEEDS`` probes, of the lag (within
    ``MAX_SHIFT_LATENCY_S``) that best aligns the energy envelopes of a seeded
    voice-like probe (see ``LATENCY_PROBE_S``) and its shifted output: pitch
    shifting moves the partials but not the envelope. Positive means the
    output is late. Cached per (filters, rate, ffmpeg executable). Raises
    ``RuntimeError`` when the lag hits the search limit (e.g. a filter that
    outputs silence).
    """
    return _shift_latency(filters, fs, ffmpeg_exe())


def _run_aligned(data: np.ndarray, fs: int, filters: str) -> np.ndarray:
    """``data`` through ``filters``, its measured delay removed, exactly ``len(data)``.

    The input is padded on both sides so that neither the head (negative delay)
    nor the tail (positive delay) of the signal is lost.
    """
    n, ch = data.shape
    lag = shift_latency(filters, fs)
    pre, post = max(0, -lag), max(0, lag) + int(round(SHIFT_FLUSH_S * fs))
    padded = np.vstack([np.zeros((pre, ch)), data, np.zeros((post, ch))])
    out = ffmpeg_float(padded, fs, filters)[pre + lag :][:n]
    if len(out) < n:
        out = np.vstack([out, np.zeros((n - len(out), ch))])
    return out


def _rubberband_filter(
    ratio: float, preserve_formants: bool, pitchq: str = DEFAULT_RUBBERBAND_PITCHQ
) -> str:
    formant = "preserved" if preserve_formants else "shifted"
    return f"rubberband=pitch={ratio:.10f}:formant={formant}:pitchq={pitchq}"


def rubberband_strategy(
    data: np.ndarray,
    fs: int,
    semitones: float,
    preserve_formants: bool,
    *,
    pitchq: str = DEFAULT_RUBBERBAND_PITCHQ,
) -> np.ndarray:
    """Pitch-shift float ``data`` with ffmpeg ``rubberband``; delay-compensated.

    The delay depends on the ratio (and a little on the signal), so it is
    measured for the actual filter by :func:`shift_latency` (e.g. +60, +22,
    -11 ms at -24, -12, +12 semitones, 44.1 kHz). On harmonic tone bursts the
    residual timing error then averages under 4 ms at 44.1/48 kHz, but the
    per-burst error from rubberband's block processing grows with the ratio:
    at the extremes (+-24 semitones) single bursts can be off by tens of ms
    (and the mean residual is larger, ~13 ms, at 16 kHz and -24).
    ``functools.partial(rubberband_strategy, pitchq="quality")`` as ``method``
    dulls less and has a smaller delay.
    """
    if pitchq not in RUBBERBAND_PITCHQ:
        raise ValueError(f"pitchq must be one of {RUBBERBAND_PITCHQ}, got {pitchq!r}")
    exe = ffmpeg_exe()
    if not has_rubberband(exe):
        raise RuntimeError(RUBBERBAND_INSTALL_HINT.format(exe=exe))
    filters = _rubberband_filter(_ratio(semitones), preserve_formants, pitchq)
    return _run_aligned(data, fs, filters)


def _atempo_chain(factor: float) -> list:
    """``atempo`` stages whose product is ``factor``, each within ``ATEMPO_RANGE``."""
    lo, hi = ATEMPO_RANGE
    stages = []
    while factor > hi or factor < lo:
        step = hi if factor > hi else lo
        stages.append(step)
        factor /= step
    return [*stages, factor]


def _resample_filter(ratio: float, fs: int) -> str:
    rate = int(round(fs * ratio))  # asetrate takes an integer rate
    tempos = _atempo_chain(fs / rate)
    return ",".join(
        [f"asetrate={rate}", f"aresample={fs}"] + [f"atempo={t:.10f}" for t in tempos]
    )


def resample_strategy(
    data: np.ndarray, fs: int, semitones: float, preserve_formants: bool
) -> np.ndarray:
    """Pitch-shift float ``data`` with stock ffmpeg filters; delay-compensated.

    ``asetrate`` (play faster or slower: pitch and tempo move), ``aresample``
    back to ``fs``, then ``atempo`` (WSOLA) restores the duration. Needs no
    optional dependency. Formants move with the pitch whatever
    ``preserve_formants`` says. Its delay is measured and removed like
    rubberband's, but ``atempo`` (WSOLA) varies more with the signal: it
    smears each burst's onset by 10-30 ms, and at +-24 semitones single
    bursts can be off by tens of ms.
    """
    return _run_aligned(data, fs, _resample_filter(_ratio(semitones), fs))


def librosa_strategy(
    data: np.ndarray, fs: int, semitones: float, preserve_formants: bool
) -> np.ndarray:
    """Pitch-shift float ``data`` with ``librosa.effects.pitch_shift``.

    Formants move with the pitch whatever ``preserve_formants`` says.
    """
    librosa = require_package("librosa")
    shifted = librosa.effects.pitch_shift(
        np.ascontiguousarray(data.T), sr=fs, n_steps=float(semitones)
    )
    return np.atleast_2d(shifted).T


def _check_method(method) -> None:
    if not callable(method) and method not in PITCH_METHODS:
        raise ValueError(
            f"method must be one of {PITCH_METHODS} or a callable, got {method!r}"
        )


def _pick_strategy(method, preserve_formants: bool) -> PitchStrategy:
    _check_method(method)
    if callable(method):
        return method
    named = {
        "rubberband": rubberband_strategy,
        "resample": resample_strategy,
        "librosa": librosa_strategy,
    }
    if method in named:
        return named[method]
    if has_rubberband():
        return rubberband_strategy
    if preserve_formants:
        warnings.warn(
            "ffmpeg has no 'rubberband' filter, so shift_pitch falls back to "
            "method='resample', which moves the formants with the pitch (the voice "
            "changes character). Pass preserve_formants=False to silence this, or "
            "install an ffmpeg with librubberband (see RUBBERBAND_INSTALL_HINT).",
            stacklevel=3,
        )
    return resample_strategy


def _fit_length(
    out: np.ndarray, n: int, channels: int, *, tolerance: int
) -> np.ndarray:
    """``out`` checked to be ``(~n, channels)`` and padded or cut to exactly ``n``."""
    if out.ndim != 2 or out.shape[1] != channels:
        raise ValueError(
            f"a pitch strategy must return float samples of shape (n, {channels}), "
            f"got shape {out.shape}"
        )
    if abs(len(out) - n) > tolerance:
        raise ValueError(
            f"a pitch strategy must keep the length: got {len(out)} frames for "
            f"{n} (tolerance {tolerance}, see LENGTH_TOLERANCE_S)"
        )
    out = out[:n]
    if len(out) < n:
        out = np.vstack([out, np.zeros((n - len(out), channels))])
    return out


def shift_pitch(
    audio: Union[str, Path, Audio],
    semitones: float,
    *,
    preserve_formants: bool = True,
    preserve_loudness: bool = True,
    method: Union[str, PitchStrategy] = "auto",
    max_semitones: float = MAX_SEMITONES,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Shift the pitch of ``audio`` by ``semitones``, keeping its duration exactly.

    Args:
        audio: path or :class:`Audio` (any ``time_unit``, kept on the result).
        semitones: shift; ``+12`` is an octave up, ``0`` returns an equal copy.
        preserve_formants: keep the vocal-tract resonances in place (only the
            rubberband method can).
        preserve_loudness: give the result the input's integrated loudness
            (BS.1770); skipped when either is silent (below the -70 LUFS gate).
        method: ``"rubberband"`` (ffmpeg with librubberband; never falls back,
            raises ``RuntimeError`` naming how to get such a build),
            ``"resample"`` (stock ffmpeg ``asetrate``/``aresample``/``atempo``,
            no optional dependency; formants move), ``"librosa"`` (extra
            ``beats``; formants move), ``"auto"`` (rubberband if available,
            else resample -- warning when ``preserve_formants``), or a
            :data:`PitchStrategy` callable, which must return ``(n, channels)``
            float samples within ``LENGTH_TOLERANCE_S`` of the input length.
        max_semitones: largest accepted ``|semitones|``.
        output: see :mod:`mixing.egress` (``None`` returns the ``Audio``).

    The result has the input's frame count, rate, channels and sample width;
    the ffmpeg methods remove their (ratio-dependent) delay. Processing is in
    float; if the shifted (and loudness-restored) signal would exceed full
    scale it is scaled down so its peak lands at ``CLIP_CEILING`` (just under
    full scale), so no clipping is introduced and the loudness drops only by
    the overshoot.
    """
    if not abs(semitones) <= max_semitones:
        raise ValueError(f"|semitones| must be <= {max_semitones}, got {semitones}")
    _check_method(method)
    audio = as_audio(audio)
    segment = audio._get_segment()
    if semitones == 0:
        result = Audio(segment._spawn(segment.raw_data), time_unit=audio.time_unit)
    else:
        strategy = _pick_strategy(method, preserve_formants)
        data = segment_samples(segment)
        fs = segment.frame_rate
        n, ch = data.shape
        out = _fit_length(
            np.asarray(strategy(data, fs, semitones, preserve_formants), dtype=float),
            n,
            ch,
            tolerance=int(round(LENGTH_TOLERANCE_S * fs)),
        )
        if preserve_loudness:
            before, after = integrated_lufs(data, fs), integrated_lufs(out, fs)
            if np.isfinite(before) and np.isfinite(after):
                out = out * 10.0 ** ((before - after) / 20.0)
        peak = float(np.max(np.abs(out), initial=0.0))
        if peak > CLIP_CEILING:
            out = out * (CLIP_CEILING / peak)
        result = to_audio(out, segment, time_unit=audio.time_unit)
    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_pitch_shifted.wav",
    )


__all__ = [
    "voice_profile",
    "shift_pitch",
    "has_rubberband",
    "shift_latency",
    "rubberband_strategy",
    "resample_strategy",
    "librosa_strategy",
]
