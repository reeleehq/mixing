"""Loudness measurement and per-turn loudness matching across speakers.

``loudness(audio)`` is an ITU-R BS.1770-4 / EBU R128 *integrated* loudness
meter (K-weighting, 400 ms blocks, absolute and relative gating) in pure
numpy/scipy, so measuring needs no ffmpeg. ``level_turns`` uses it to bring
every speaker turn of a recording to the same loudness, which whole-track peak
normalisation (:meth:`Audio.normalize`) cannot do. ``loudness_range`` (EBU
Tech 3342 LRA) and ``true_peak_db`` (4x oversampled) sit beside it.

``level_turns`` is :func:`mixing.audio.turns.process_turns` with a chain of
stages per turn, all run on the turn padded by ``pad_s`` (so filters settle):
the speaker's EQ -> pre-gain to a common level -> compressor -> gain to
``target_lufs`` -> denoise when the noise floor is high -> short fades -> peak
limiter at ``ceiling_db``. The chain runs on **float** samples (ffmpeg filters
are piped as 64-bit float PCM), so gains carry headroom above full scale and
nothing is clipped before the limiter; the turn is converted back to the
input's integer sample width only at the end. Filters are ffmpeg filter
strings (the ``chains``/``compressor``/``denoise`` seams also accept any
``Audio -> Audio`` callable, which sees integer PCM and so saturates anything
above full scale).

>>> from pydub import AudioSegment
>>> loudness(Audio(AudioSegment.silent(1000)))
-inf
"""

from __future__ import annotations

import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe
from .audio_ops import _MIX_SILENCE_FLOOR_DB, Audio
from .samples import as_audio, ffmpeg_float, audio_samples, segment_samples, to_audio
from .turns import DEFAULT_PAD_S, Core, Turn, conform_segment, process_turns

AudioFilter = Union[str, Callable[[Audio], Audio]]  # ffmpeg filter string or callable
#: A stage of the per-turn chain: float samples ``(n, channels)`` in, same out.
FloatStage = Callable[[np.ndarray], np.ndarray]

# --- BS.1770-4 constants -----------------------------------------------------
BLOCK_S = 0.4
BLOCK_HOP_S = 0.1
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
_LUFS_OFFSET = -0.691
_SHELF = dict(f0=1681.974450955533, gain_db=3.999843853973347, q=0.7071752369554196)
_SHELF_VB_EXPONENT = 0.4996667741545416
_HIGHPASS = dict(f0=38.13547087602444, q=0.5003270373238773)

#: Seconds filtered at a time when metering (bounds peak memory).
FILTER_CHUNK_S = 60.0
# EBU Tech 3342 loudness range: 3 s short-term windows, relative gate -20 LU,
# the spread between the 10th and 95th percentile of the gated windows.
LRA_WINDOW_S = 3.0
LRA_HOP_S = 1.0
LRA_RELATIVE_GATE_LU = -20.0
LRA_PERCENTILES = (10.0, 95.0)
#: True-peak (BS.1770 Annex 2) oversampling factor.
TRUE_PEAK_OVERSAMPLING = 4
#: The true-peak interpolation is meant for >= 48 kHz; below that the factor is
#: raised so the oversampled rate reaches this.
TRUE_PEAK_MIN_RATE = 192_000
#: Context (s) shared by neighbouring chunks so the upsampling filter does not
#: ring at chunk joins.
TRUE_PEAK_MARGIN_S = 0.1

# --- level_turns defaults ----------------------------------------------------
DEFAULT_TARGET_LUFS = -16.0
#: Level every turn is first brought to, so the compressor treats all voices alike.
DEFAULT_PREGAIN_LUFS = -18.0
DEFAULT_COMPRESSOR = "acompressor=threshold=-20dB:ratio=2:attack=10:release=160:knee=6"
DEFAULT_DENOISE = "afftdn=nr=10:nf=-50"
#: Denoise a turn whose noise floor (below) sits above this dBFS.
DEFAULT_DENOISE_ABOVE_DBFS = -58.0
DEFAULT_FADE_S = 0.02
DEFAULT_CEILING_DB = -1.5
#: Accepted ``ceiling_db`` range (dBFS); ffmpeg's ``alimiter`` takes limits
#: from 0.0625 (-24.1 dBFS) to 1 (0 dBFS).
CEILING_DB_RANGE = (-24.0, 0.0)
#: Most a turn's loudness may be raised (dB). A turn needing more is mostly
#: silence or room tone, and boosting it further only amplifies noise; it is
#: capped here and a warning names it.
MAX_GAIN_DB = 24.0
NOISE_FRAME_S = 0.05
NOISE_PERCENTILE = 10

# --- ffmpeg filter latency ---------------------------------------------------
#: Longest filter latency (s) ``apply_ffmpeg_filter`` searches for and removes
#: (``afftdn`` delays by ~25 ms).
MAX_FILTER_LATENCY_S = 0.1
#: Length (s), level and seed of the white-noise probe the latency is measured
#: on. Latency is a property of the filter, not of the signal, so it is
#: measured once per (filter, rate, channels) on broadband noise, whose
#: correlation has one sharp peak -- tonal or hummy programme material would
#: give a periodic correlation with several near-equal peaks.
LATENCY_PROBE_S = 1.0
LATENCY_PROBE_AMPLITUDE = 0.1
LATENCY_PROBE_SEED = 1770
#: The probe's head (fraction) is skipped while an adaptive filter settles.
LATENCY_SETTLE_FRACTION = 0.25
#: A lag is accepted only when its normalised correlation is at least this...
LATENCY_MIN_CORRELATION = 0.5
#: ...and at least this much above the correlation at lag 0; otherwise 0.
LATENCY_MIN_ADVANTAGE = 0.05


# --- measurement ----------------------------------------------------------------


def _k_weighting_coefficients(fs: int):
    """``[(b, a), (b, a)]`` biquads of the BS.1770 K-weighting at rate ``fs``."""
    k = np.tan(np.pi * _SHELF["f0"] / fs)
    vh = 10.0 ** (_SHELF["gain_db"] / 20.0)
    vb = vh**_SHELF_VB_EXPONENT
    q = _SHELF["q"]
    a0 = 1.0 + k / q + k * k
    shelf = (
        np.array([vh + vb * k / q + k * k, 2 * (k * k - vh), vh - vb * k / q + k * k]) / a0,
        np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]),
    )  # fmt: skip
    k = np.tan(np.pi * _HIGHPASS["f0"] / fs)
    q = _HIGHPASS["q"]
    a0 = 1.0 + k / q + k * k
    highpass = (
        np.array([1.0, -2.0, 1.0]),
        np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]),
    )
    return [shelf, highpass]


def _lufs(mean_square: float) -> float:
    return _LUFS_OFFSET + 10.0 * np.log10(mean_square) if mean_square > 0 else -np.inf


def _k_weighted_energy(data: np.ndarray, fs: int, *, chunk_s: float) -> np.ndarray:
    """Channel-summed squared K-weighted samples, ``(n,)``.

    Filters ``chunk_s`` seconds at a time, carrying the filter state across
    chunks, so peak memory is a few chunks of float64 rather than several
    copies of the whole (possibly hours-long, stereo) signal.
    """
    from scipy.signal import lfilter

    sections = _k_weighting_coefficients(fs)
    states = [np.zeros((2, data.shape[1])) for _ in sections]  # biquads: order 2
    energy = np.empty(len(data))
    step = max(1, int(round(chunk_s * fs)))
    for start in range(0, len(data), step):
        x = data[start : start + step]
        for k, (b, a) in enumerate(sections):
            x, states[k] = lfilter(b, a, x, axis=0, zi=states[k])
        energy[start : start + step] = (x**2).sum(axis=1)
    return energy


def _block_mean_squares(
    energy: np.ndarray, fs: int, *, block_s: float, hop_s: float
) -> np.ndarray:
    """Mean square of every ``block_s`` window at ``hop_s`` spacing (one block if short)."""
    block, hop = int(round(block_s * fs)), int(round(hop_s * fs))
    if len(energy) < block:
        block, hop = len(energy), len(energy)
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    starts = np.arange(0, len(energy) - block + 1, hop)
    return (cumulative[starts + block] - cumulative[starts]) / block


def _gated(block_ms: np.ndarray, relative_gate_lu: float):
    """``(block_ms, block_lufs)`` surviving the absolute and relative gates."""
    with np.errstate(divide="ignore"):
        block_lufs = _LUFS_OFFSET + 10.0 * np.log10(block_ms)  # -inf where silent
    absolute = block_lufs > ABSOLUTE_GATE_LUFS
    if not absolute.any():
        return block_ms[:0], block_lufs[:0]
    relative_gate = _lufs(block_ms[absolute].mean()) + relative_gate_lu
    keep = absolute & (block_lufs > relative_gate)
    return block_ms[keep], block_lufs[keep]


def integrated_lufs(
    data: np.ndarray, fs: int, *, chunk_s: float = FILTER_CHUNK_S
) -> float:
    """BS.1770-4 gated integrated loudness of float samples ``(n, channels)``."""
    if len(data) == 0:
        return -np.inf
    energy = _k_weighted_energy(data, fs, chunk_s=chunk_s)
    block_ms = _block_mean_squares(energy, fs, block_s=BLOCK_S, hop_s=BLOCK_HOP_S)
    kept, _ = _gated(block_ms, RELATIVE_GATE_LU)
    return float(_lufs(kept.mean())) if len(kept) else -np.inf


def loudness(audio: Union[str, Path, Audio]) -> float:
    """Integrated loudness of ``audio`` in LUFS (ITU-R BS.1770-4, gated).

    Returns ``-inf`` for silence (nothing above the -70 LUFS absolute gate).
    Every channel is weighted 1 (right for mono and stereo). A clip shorter
    than one 400 ms block is measured as a single block.
    """
    return integrated_lufs(*audio_samples(as_audio(audio)))


def loudness_range(audio: Union[str, Path, Audio]) -> float:
    """Loudness range (LRA) of ``audio`` in LU (EBU Tech 3342).

    The spread between the 10th and 95th percentile of the gated 3 s
    short-term loudness. 0.0 for silence or a clip shorter than 3 s.
    """
    data, fs = audio_samples(as_audio(audio))
    if len(data) == 0:
        return 0.0
    energy = _k_weighted_energy(data, fs, chunk_s=FILTER_CHUNK_S)
    if len(energy) < round(LRA_WINDOW_S * fs):
        return 0.0
    windows = _block_mean_squares(energy, fs, block_s=LRA_WINDOW_S, hop_s=LRA_HOP_S)
    _, lufs = _gated(windows, LRA_RELATIVE_GATE_LU)
    if len(lufs) == 0:
        return 0.0
    low, high = np.percentile(lufs, LRA_PERCENTILES)
    return float(high - low)


def true_peak_db(
    audio: Union[str, Path, Audio], *, oversampling: Optional[int] = None
) -> float:
    """True peak of ``audio`` in dBTP: the sample peak after upsampling.

    Catches inter-sample peaks a sample-peak meter misses. ``oversampling``
    defaults to 4x, raised for rates under 48 kHz so the oversampled rate is at
    least ``TRUE_PEAK_MIN_RATE``. ``-inf`` for silence.
    """
    from scipy.signal import resample_poly

    data, fs = audio_samples(as_audio(audio))
    if len(data) == 0:
        return -np.inf
    factor = oversampling or max(TRUE_PEAK_OVERSAMPLING, -(-TRUE_PEAK_MIN_RATE // fs))
    step = max(1, int(round(FILTER_CHUNK_S * fs)))
    margin = int(round(TRUE_PEAK_MARGIN_S * fs))
    peak = 0.0
    for start in range(0, len(data), step):
        stop = min(start + step, len(data))
        lo, hi = max(0, start - margin), min(len(data), stop + margin)
        up = resample_poly(data[lo:hi], factor, 1, axis=0)
        peak = max(
            peak, float(np.abs(up[(start - lo) * factor : (stop - lo) * factor]).max())
        )
    return float(20.0 * np.log10(peak)) if peak > 0 else -np.inf


def _noise_floor(
    data: np.ndarray, fs: int, *, frame_s: float, percentile: float
) -> float:
    mono = data.mean(axis=1)
    frame_n = max(1, int(round(frame_s * fs)))
    n_frames = max(1, int(np.ceil(len(mono) / frame_n)))
    padded = np.zeros(n_frames * frame_n)
    padded[: len(mono)] = mono
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame_n) ** 2, axis=1))
    floor = 10.0 ** (_MIX_SILENCE_FLOOR_DB / 20.0)
    return float(np.percentile(20.0 * np.log10(np.maximum(rms, floor)), percentile))


def noise_floor_db(
    audio: Audio,
    *,
    frame_s: float = NOISE_FRAME_S,
    percentile: float = NOISE_PERCENTILE,
) -> float:
    """Noise floor estimate: a low percentile of the ``frame_s`` RMS levels (dBFS)."""
    return _noise_floor(*audio_samples(audio), frame_s=frame_s, percentile=percentile)


def apply_gain(audio: Audio, db: float) -> Audio:
    """``audio`` with ``db`` decibels of gain, computed in float.

    The result is converted back to ``audio``'s integer sample width, so peaks
    pushed beyond full scale saturate; :func:`level_turns` keeps its gains in
    float and only converts after its limiter.
    """
    data, _ = audio_samples(audio)
    gained = data * 10.0 ** (db / 20.0)
    return to_audio(gained, audio._get_segment(), time_unit=audio.time_unit)


# --- ffmpeg filters -----------------------------------------------------------


@lru_cache(maxsize=64)
def filter_latency(filters: str, fs: int, channels: int, *, max_lag: int) -> int:
    """Delay (samples, ``0..max_lag``) ffmpeg ``filters`` add at rate ``fs``.

    Measured on a seeded white-noise probe (see ``LATENCY_PROBE_S``), by
    normalised cross-correlation; 0 unless the best lag clearly wins
    (``LATENCY_MIN_CORRELATION``, ``LATENCY_MIN_ADVANTAGE`` over lag 0).
    Cached per (filters, rate, channels, max_lag).
    """
    if max_lag <= 0:
        return 0
    n = int(round(LATENCY_PROBE_S * fs))
    rng = np.random.default_rng(LATENCY_PROBE_SEED)
    probe = LATENCY_PROBE_AMPLITUDE * rng.standard_normal((n, channels))
    out = ffmpeg_float(np.vstack([probe, np.zeros((max_lag, channels))]), fs, filters)
    ref, res = probe.mean(axis=1), out.mean(axis=1)
    lo = int(n * LATENCY_SETTLE_FRACTION)
    ref = ref[lo:]
    res = np.concatenate(
        [res[lo:], np.zeros(max(0, len(ref) + max_lag - len(res[lo:])))]
    )
    size = 1 << int(np.ceil(np.log2(len(res) + len(ref))))
    xcorr = np.fft.irfft(np.fft.rfft(res, size) * np.conj(np.fft.rfft(ref, size)), size)
    xcorr = xcorr[: max_lag + 1]
    energy = np.concatenate([[0.0], np.cumsum(res**2)])
    lags = np.arange(max_lag + 1)
    window_energy = energy[lags + len(ref)] - energy[lags]
    norm = np.sqrt(np.maximum(window_energy, 1e-30) * np.sum(ref**2))
    corr = xcorr / norm
    lag = int(np.argmax(corr))
    if (
        corr[lag] < LATENCY_MIN_CORRELATION
        or corr[lag] - corr[0] < LATENCY_MIN_ADVANTAGE
    ):
        return 0
    return lag


def _run_filter(
    data: np.ndarray,
    fs: int,
    filters: str,
    *,
    compensate_latency: bool,
    max_latency_s: float = MAX_FILTER_LATENCY_S,
) -> np.ndarray:
    """Float ``data`` through ``filters``, same length; optionally delay-compensated."""
    n, ch = data.shape
    lag = 0
    if compensate_latency:
        max_lag = int(round(max_latency_s * fs))
        lag = filter_latency(filters, fs, ch, max_lag=max_lag)
    out = ffmpeg_float(np.vstack([data, np.zeros((lag, ch))]), fs, filters)[lag:]
    if len(out) < n:
        out = np.vstack([out, np.zeros((n - len(out), ch))])
    return out[:n]


def apply_ffmpeg_filter(
    audio: Audio,
    filters: str,
    *,
    compensate_latency: bool = False,
    max_latency_s: float = MAX_FILTER_LATENCY_S,
) -> Audio:
    """Run an ffmpeg ``-af`` filter chain over ``audio``; returns a new Audio.

    Keeps the sample rate, channel count, sample width and length. Audio is
    piped as float PCM (no temp files), so the filter itself never clips; the
    result saturates at full scale when converted back to integer samples.

    ``compensate_latency`` is for filters with a fixed processing delay (e.g.
    ``afftdn``'s ~25 ms): the delay is measured once on a noise probe (see
    :func:`filter_latency`, up to ``max_latency_s``) and removed, so onsets
    stay put. Leave it off for filters without one (EQ, compressors).
    """
    data, fs = audio_samples(audio)
    out = _run_filter(
        data,
        fs,
        filters,
        compensate_latency=compensate_latency,
        max_latency_s=max_latency_s,
    )
    return to_audio(out, audio._get_segment(), time_unit=audio.time_unit)


def _as_stage(
    spec: AudioFilter | None, fs: int, like, *, compensate_latency: bool = False
) -> FloatStage | None:
    """A float stage from an ffmpeg filter string or an ``Audio -> Audio`` callable."""
    if spec is None:
        return None
    if callable(spec):

        def via_audio(x: np.ndarray) -> np.ndarray:
            out = spec(to_audio(x, like))  # integer PCM: saturates above 0 dBFS
            return segment_samples(conform_segment(out._get_segment(), like))

        return via_audio
    return lambda x: _run_filter(x, fs, spec, compensate_latency=compensate_latency)


# --- level_turns stages -------------------------------------------------------


def _fade(data: np.ndarray, fs: int, core: Core, fade_s: float) -> np.ndarray:
    """Linear fade in at ``core[0]`` and out at ``core[1]`` (capped to half the turn)."""
    b = min(len(data), int(round(core[1] * fs)))  # a stage may have shortened it
    a = min(b, int(round(core[0] * fs)))
    n = min(int(round(fade_s * fs)), (b - a) // 2)
    if n <= 0:
        return data
    ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    data = data.copy()
    data[a : a + n] *= ramp
    data[b - n : b] *= ramp[::-1]
    return data


def _chain(stages: Sequence[FloatStage | None]) -> FloatStage:
    """Compose stages left to right; ``None`` stages are skipped."""

    def run(data: np.ndarray) -> np.ndarray:
        for stage in stages:
            if stage is not None:
                data = stage(data)
        return data

    return run


def _check_ceiling(ceiling_db: float | None) -> None:
    lo, hi = CEILING_DB_RANGE
    if ceiling_db is not None and not lo <= ceiling_db <= hi:
        raise ValueError(f"ceiling_db must be in [{lo}, {hi}] dBFS, got {ceiling_db}")


def level_turns(
    audio: Union[str, Path, Audio],
    turns: Sequence[Turn],
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    chains: Mapping[str, AudioFilter] | None = None,
    pregain_lufs: float = DEFAULT_PREGAIN_LUFS,
    compressor: AudioFilter | None = DEFAULT_COMPRESSOR,
    denoise: AudioFilter | None = DEFAULT_DENOISE,
    denoise_above: float = DEFAULT_DENOISE_ABOVE_DBFS,
    pad_s: float = DEFAULT_PAD_S,
    fade_s: float = DEFAULT_FADE_S,
    max_gain_db: float = MAX_GAIN_DB,
    ceiling_db: float | None = DEFAULT_CEILING_DB,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Bring every speaker turn of ``audio`` to the same loudness.

    Each turn is processed padded by ``pad_s`` (see
    :func:`mixing.audio.turns.process_turns`), then trimmed back to its exact
    length, so the output has the input's duration.

    Args:
        audio: the recording (path or :class:`Audio`; any ``time_unit``).
        turns: ``[(speaker, start_s, end_s), ...]`` in seconds; every boundary
            should lie in a pause.
        target_lufs: integrated loudness every turn ends at.
        chains: ``{speaker: ffmpeg filter string | Audio -> Audio}`` per-speaker
            EQ. Speakers without an entry get no EQ.
        pregain_lufs: level each turn is set to before the compressor, so it
            treats all voices alike.
        compressor: ffmpeg filter / callable / ``None`` to skip.
        denoise: filter applied to a turn whose noise floor (10th percentile of
            50 ms RMS), after levelling, is above ``denoise_above`` dBFS;
            ``None`` to skip. A filter string's fixed delay is compensated.
        pad_s: context kept around each turn while filtering.
        fade_s: fade at each turn boundary, hiding the seam.
        max_gain_db: most a turn's loudness may be raised; beyond it the gain
            is capped and a warning is issued.
        ceiling_db: peak limit of every turn in dBFS, in
            ``CEILING_DB_RANGE`` (a lookahead limiter, no auto-level,
            latency-compensated); ``None`` to skip -- then any peak the gains
            push above full scale **saturates** when the turn is written back
            to integer PCM.
        output: see :mod:`mixing.egress` (``None`` returns the ``Audio``).

    Notes:
        - Audio **between** turns is left untouched (not even limited). Room
          tone *inside* a turn is gained with the voice, so the noise floor
          can step at turn boundaries.
        - A turn shorter than 400 ms is measured as one block (BS.1770 needs
          at least one full block for gating).
        - A non-silent turn entirely below the -70 LUFS absolute gate cannot
          be measured; it is left untouched, with a warning.
        - Gains, ffmpeg filters, fades and the limiter run on float samples;
          a *callable* stage gets integer PCM and saturates above full scale.

    Returns:
        The levelled :class:`Audio` (or the saved ``Path``).
    """
    _check_ceiling(ceiling_db)
    chains = chains or {}
    limiter = None
    if ceiling_db is not None:
        limit = 10.0 ** (ceiling_db / 20.0)
        # level=0: no auto make-up gain (the default raises output to 0 dBFS);
        # latency=1: alimiter compensates its own lookahead delay.
        limiter = f"alimiter=limit={limit:.6f}:level=0:latency=1"

    def process(speaker: str, padded: Audio, core: Core) -> Audio:
        like = padded._get_segment()
        data, fs = segment_samples(like), like.frame_rate
        a, b = int(round(core[0] * fs)), int(round(core[1] * fs))

        def measure(x: np.ndarray) -> float:
            return integrated_lufs(x[a:b], fs)

        def gain_to(target_db: float) -> FloatStage:
            def stage(x: np.ndarray) -> np.ndarray:
                current = measure(x)
                return (
                    x
                    if not np.isfinite(current)
                    else x * 10.0 ** ((target_db - current) / 20.0)
                )

            return stage

        clean = _as_stage(denoise, fs, like, compensate_latency=True)

        def denoise_if_noisy(x: np.ndarray) -> np.ndarray:
            floor = _noise_floor(
                x[a:b], fs, frame_s=NOISE_FRAME_S, percentile=NOISE_PERCENTILE
            )
            return x if floor <= denoise_above else gain_to(target)(clean(x))

        eq = _as_stage(chains.get(speaker), fs, like)
        data = _chain([eq])(data)
        original = measure(data)
        if not np.isfinite(original):
            if np.any(data[a:b]):
                warnings.warn(
                    f"turn {speaker!r} at {core[0]:.2f}s of its clip is below the "
                    f"{ABSOLUTE_GATE_LUFS} LUFS gate and was left untouched",
                    stacklevel=3,
                )
            return padded
        target, pregain = target_lufs, pregain_lufs
        if target - original > max_gain_db:
            warnings.warn(
                f"turn {speaker!r} at {original:.1f} LUFS needs "
                f"{target - original:.1f} dB to reach {target} LUFS; "
                f"gain capped at {max_gain_db} dB",
                stacklevel=3,
            )
            target = original + max_gain_db
            pregain = min(pregain, target)
        data = _chain(
            [
                gain_to(pregain),
                _as_stage(compressor, fs, like),
                gain_to(target),
                denoise_if_noisy if clean else None,
                lambda x: _fade(x, fs, core, fade_s),
                _as_stage(limiter, fs, like),
            ]
        )(data)
        return to_audio(data, like)

    result = process_turns(audio, turns, process, pad_s=pad_s)
    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_leveled.wav",
    )
