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
   exactly. ``method="rubberband"`` uses ffmpeg's ``rubberband`` filter, which can
   **preserve formants** (the voice stays the same person, just higher/lower);
   it needs an ffmpeg built with ``librubberband``. ``method="librosa"``
   (phase vocoder + resampling, extra ``beats``) moves formants with the pitch.
   ``method="auto"`` prefers rubberband and falls back to librosa with a warning
   when formants were asked to be preserved.

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
from math import gcd
from pathlib import Path
from typing import Callable, Hashable, Mapping, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe
from .audio_ops import Audio
from .leveling import (
    _ffmpeg_float,
    _integrated_lufs,
    _segment_samples,
    _to_audio,
    filter_latency,
)
from .turns import TURN_BOUND_TOLERANCE_S

Label = Hashable
Span = tuple[float, float]  # (start_s, end_s)
#: ``[(label, start_s, end_s), ...]`` (the ``turns`` shape of ``level_turns``) or
#: ``{label: [(start_s, end_s), ...]}``.
Spans = Union[Sequence[tuple[Label, float, float]], Mapping[Label, Sequence[Span]]]
#: ``strategy(samples (n, channels) float, rate, semitones, preserve_formants) -> samples``
PitchStrategy = Callable[[np.ndarray, int, float, bool], np.ndarray]

BEATS_INSTALL_HINT = "pip install 'mixing[beats]'"

# --- voice_profile defaults ---------------------------------------------------
#: pyin search range (Hz): covers low male (~65 Hz) to high female/child speech.
DEFAULT_F0_FMIN_HZ = 65.0
DEFAULT_F0_FMAX_HZ = 400.0
#: Rate (Hz) the f0 tracker runs at; ample for f0 <= 400 Hz and keeps pyin fast.
DEFAULT_F0_SAMPLE_RATE = 16000
#: pyin frame (samples at ``DEFAULT_F0_SAMPLE_RATE``): 128 ms, > 8 periods of fmin.
DEFAULT_F0_FRAME_LENGTH = 2048
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
PITCH_METHODS = ("auto", "rubberband", "librosa")
#: Longest rubberband processing delay (s) searched for and removed.
MAX_RUBBERBAND_LATENCY_S = 0.5
RUBBERBAND_INSTALL_HINT = (
    "ffmpeg's 'rubberband' filter is not available in {exe!r}. Install an ffmpeg "
    "built with --enable-librubberband (macOS: `brew tap homebrew-ffmpeg/ffmpeg && "
    "brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-rubberband`; Debian/Ubuntu: "
    "`apt install ffmpeg`, whose build includes it) and point mixing at it with "
    "the IMAGEIO_FFMPEG_EXE environment variable, or use method='librosa' (which "
    "does not preserve formants)."
)


def _require_librosa(purpose: str):
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            f"librosa is required for {purpose}. Install it via {BEATS_INSTALL_HINT!r}."
        ) from e
    return librosa


def _as_audio(audio: Union[str, Path, Audio]) -> Audio:
    return audio if isinstance(audio, Audio) else Audio(audio)


# --- spans --------------------------------------------------------------------


def _spans_by_label(spans: Optional[Spans], duration: float) -> dict:
    """``{label: [(start, end), ...]}`` merged (union), sorted, range-checked."""
    if spans is None:
        grouped = {WHOLE_AUDIO_LABEL: [(0.0, duration)]}
    elif isinstance(spans, Mapping):
        grouped = {label: list(ranges) for label, ranges in spans.items()}
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


def _resample(mono: np.ndarray, fs: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if fs == target:
        return mono
    g = gcd(fs, target)
    return resample_poly(mono, target // g, fs // g)


def _voiced_f0(
    pieces: Sequence[np.ndarray],
    fs: int,
    *,
    fmin: float,
    fmax: float,
    sample_rate: int,
    frame_length: int,
    min_voiced_prob: float,
) -> tuple[Optional[float], float]:
    """``(median f0 over voiced frames or None, voiced fraction)`` of mono ``pieces``.

    Each piece (one span) is tracked separately, so span seams make no false
    pitch; pieces shorter than one frame are skipped.
    """
    librosa = _require_librosa("f0 estimation (voice_profile)")
    voiced, n_frames = [], 0
    for piece in pieces:
        y = _resample(piece, fs, sample_rate)
        if len(y) < frame_length or not np.any(y):
            continue
        f0, flags, probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=frame_length
        )
        n_frames += len(f0)
        voiced.append(f0[flags & (probs >= min_voiced_prob) & np.isfinite(f0)])
    values = np.concatenate(voiced) if voiced else np.array([])
    fraction = len(values) / n_frames if n_frames else 0.0
    return (float(np.median(values)) if len(values) else None), float(fraction)


def voice_profile(
    audio: Union[str, Path, Audio],
    spans: Optional[Spans] = None,
    *,
    f0: bool = True,
    fmin: float = DEFAULT_F0_FMIN_HZ,
    fmax: float = DEFAULT_F0_FMAX_HZ,
    f0_sample_rate: int = DEFAULT_F0_SAMPLE_RATE,
    f0_frame_length: int = DEFAULT_F0_FRAME_LENGTH,
    min_voiced_prob: float = DEFAULT_MIN_VOICED_PROB,
    low_band: Span = DEFAULT_LOW_BAND_HZ,
    welch_nperseg: int = DEFAULT_WELCH_NPERSEG,
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
        f0: estimate f0 (needs librosa, extra ``beats``); ``False`` skips it
            (``f0_hz`` and ``voiced_fraction`` are then ``None``).
        fmin, fmax: pyin f0 search range (Hz).
        f0_sample_rate, f0_frame_length: the f0 tracker's rate and frame size.
        min_voiced_prob: pyin voicing probability below which a frame counts
            as unvoiced (keeps noise from producing a spurious f0).
        low_band: ``(lo_hz, hi_hz)`` of the band reported by ``low_band_db``.
        welch_nperseg: Welch segment length for centroid and low-band ratio.

    Returns:
        ``{label: {...}}`` of plain JSON-able dicts with keys

        - ``duration_s``: total length of the label's (merged) spans;
        - ``f0_hz``: median f0 over voiced frames (``None`` if none are voiced);
        - ``voiced_fraction``: share of pyin frames judged voiced;
        - ``loudness_lufs``: BS.1770 integrated loudness of the union
          (``None`` when below the -70 LUFS gate, i.e. silence);
        - ``spectral_centroid_hz``: power-weighted mean frequency;
        - ``low_band_db``: ``10 log10`` of the ``low_band`` energy over the
          full-band energy (higher = boomier; ``None`` for silence).
    """
    audio = _as_audio(audio)
    segment = audio._get_segment()
    fs = segment.frame_rate
    data = _segment_samples(segment)
    grouped = _spans_by_label(spans, len(data) / fs)

    def profile(ranges) -> dict:
        frames = [
            (int(round(s * fs)), min(len(data), int(round(e * fs)))) for s, e in ranges
        ]
        pieces = [data[a:b] for a, b in frames if b > a]
        union = np.concatenate(pieces) if pieces else np.zeros((0, data.shape[1]))
        mono = union.mean(axis=1)
        lufs = _integrated_lufs(union, fs)
        centroid, low_db = _band_stats(
            mono, fs, low_band=low_band, nperseg=welch_nperseg
        )
        f0_hz = fraction = None
        if f0:
            f0_hz, fraction = _voiced_f0(
                [p.mean(axis=1) for p in pieces],
                fs,
                fmin=fmin,
                fmax=fmax,
                sample_rate=f0_sample_rate,
                frame_length=f0_frame_length,
                min_voiced_prob=min_voiced_prob,
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


@lru_cache(maxsize=8)
def _ffmpeg_filters(exe: str) -> frozenset:
    """Names of the filters ``exe`` was built with (probed once per executable)."""
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-filters"], capture_output=True, text=True
        )
    except OSError:
        return frozenset()
    names = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_rubberband(ffmpeg: Optional[str] = None) -> bool:
    """Whether ``ffmpeg`` (default :func:`mixing.util.ffmpeg_exe`) has ``rubberband``."""
    return "rubberband" in _ffmpeg_filters(ffmpeg or ffmpeg_exe())


def _rubberband_filter(ratio: float, preserve_formants: bool) -> str:
    formant = "preserved" if preserve_formants else "shifted"
    return f"rubberband=pitch={ratio:.10f}:formant={formant}"


def rubberband_strategy(
    data: np.ndarray, fs: int, semitones: float, preserve_formants: bool
) -> np.ndarray:
    """Pitch-shift float ``data`` with ffmpeg ``rubberband``; delay-compensated.

    Rubberband's delay is measured on its identity setting (``pitch=1``, same
    formant mode), since a shifted output does not correlate with its input.
    """
    exe = ffmpeg_exe()
    if not has_rubberband(exe):
        raise RuntimeError(RUBBERBAND_INSTALL_HINT.format(exe=exe))
    n, ch = data.shape
    max_lag = int(round(MAX_RUBBERBAND_LATENCY_S * fs))
    lag = filter_latency(
        _rubberband_filter(1.0, preserve_formants), fs, ch, max_lag=max_lag
    )
    ratio = 2.0 ** (semitones / SEMITONES_PER_OCTAVE)
    filters = _rubberband_filter(ratio, preserve_formants)
    return _ffmpeg_float(np.vstack([data, np.zeros((lag, ch))]), fs, filters)[lag:]


def librosa_strategy(
    data: np.ndarray, fs: int, semitones: float, preserve_formants: bool
) -> np.ndarray:
    """Pitch-shift float ``data`` with ``librosa.effects.pitch_shift``.

    Formants move with the pitch whatever ``preserve_formants`` says.
    """
    librosa = _require_librosa("pitch shifting with method='librosa'")
    shifted = librosa.effects.pitch_shift(
        np.ascontiguousarray(data.T), sr=fs, n_steps=float(semitones)
    )
    return np.atleast_2d(shifted).T


def _pick_strategy(method, preserve_formants: bool) -> PitchStrategy:
    if callable(method):
        return method
    if method not in PITCH_METHODS:
        raise ValueError(
            f"method must be one of {PITCH_METHODS} or a callable, got {method!r}"
        )
    if method == "rubberband":
        return rubberband_strategy
    if method == "librosa":
        return librosa_strategy
    if has_rubberband():
        return rubberband_strategy
    if preserve_formants:
        warnings.warn(
            "ffmpeg has no 'rubberband' filter, so shift_pitch falls back to librosa, "
            "which moves the formants with the pitch (the voice changes character). "
            "Pass preserve_formants=False to silence this, or install an ffmpeg with "
            "librubberband (see RUBBERBAND_INSTALL_HINT).",
            stacklevel=3,
        )
    return librosa_strategy


def _fit_length(out: np.ndarray, n: int, channels: int) -> np.ndarray:
    """``out`` zero-padded or cut to exactly ``n`` frames."""
    out = out.reshape(-1, channels)[:n]
    if len(out) < n:
        out = np.vstack([out, np.zeros((n - len(out), channels))])
    return out


def shift_pitch(
    audio: Union[str, Path, Audio],
    semitones: float,
    *,
    preserve_formants: bool = True,
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
        method: ``"rubberband"`` (ffmpeg with librubberband; never falls back,
            raises ``RuntimeError`` naming how to get such a build),
            ``"librosa"`` (extra ``beats``; formants move), ``"auto"``
            (rubberband if available, else librosa -- warning when
            ``preserve_formants``), or a :data:`PitchStrategy` callable.
        max_semitones: largest accepted ``|semitones|``.
        output: see :mod:`mixing.egress` (``None`` returns the ``Audio``).

    The result has the input's frame count, rate, channels and sample width.
    Processing is in float; if the shifted signal would exceed full scale it is
    scaled down to the input's peak, so no clipping is introduced.
    """
    if not abs(semitones) <= max_semitones:
        raise ValueError(f"|semitones| must be <= {max_semitones}, got {semitones}")
    audio = _as_audio(audio)
    segment = audio._get_segment()
    if semitones == 0:
        result = Audio(segment._spawn(segment.raw_data), time_unit=audio.time_unit)
    else:
        strategy = _pick_strategy(method, preserve_formants)
        data = _segment_samples(segment)
        n, ch = data.shape
        out = _fit_length(
            np.asarray(
                strategy(data, segment.frame_rate, semitones, preserve_formants)
            ),
            n,
            ch,
        )
        peak, in_peak = (
            float(np.max(np.abs(out), initial=0.0)),
            float(np.max(np.abs(data), initial=0.0)),
        )
        if peak > 1.0:
            out = out * (in_peak / peak)
        result = _to_audio(out, segment, time_unit=audio.time_unit)
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
    "rubberband_strategy",
    "librosa_strategy",
]
