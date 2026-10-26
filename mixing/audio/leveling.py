"""Loudness measurement and per-turn loudness matching across speakers.

``loudness(audio)`` is an ITU-R BS.1770-4 / EBU R128 *integrated* loudness
meter (K-weighting, 400 ms blocks, absolute and relative gating) in pure
numpy/scipy, so measuring needs no ffmpeg. ``level_turns`` uses it to bring
every speaker turn of a recording to the same loudness, which whole-track peak
normalisation (:meth:`Audio.normalize`) cannot do.

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
from typing import Callable, Mapping, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe
from .audio_ops import _MIX_SILENCE_FLOOR_DB, Audio
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

_FLOAT_PCM = "f64le"
_INT_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}  # pydub stores 8-bit signed


# --- samples <-> Audio ----------------------------------------------------------


def _segment_samples(segment) -> np.ndarray:
    """Float samples ``(n, channels)`` in [-1, 1] of a pydub segment."""
    data = np.array(segment.get_array_of_samples(), dtype=np.float64)
    return data.reshape(-1, segment.channels) / float(
        1 << (8 * segment.sample_width - 1)
    )


def _samples(audio: Audio) -> tuple[np.ndarray, int]:
    """``(float samples shaped (n, channels) in [-1, 1], rate)`` of ``audio``."""
    segment = audio._get_segment()
    return _segment_samples(segment), segment.frame_rate


def _to_audio(data: np.ndarray, like, *, time_unit: str = "seconds") -> Audio:
    """Float ``data`` as an :class:`Audio` with ``like``'s rate, channels, width.

    Values beyond full scale **saturate** (integer PCM has no headroom).
    """
    width = like.sample_width
    full_scale = float(1 << (8 * width - 1))
    ints = np.clip(np.round(data * full_scale), -full_scale, full_scale - 1)
    if width == 3:  # pydub converts 24-bit to 32-bit, but be safe
        raw = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raw = ints.astype(_INT_DTYPES[width]).tobytes()
    return Audio(like._spawn(raw), time_unit=time_unit)


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


def _integrated_lufs(data: np.ndarray, fs: int) -> float:
    """BS.1770-4 gated integrated loudness of float samples ``(n, channels)``."""
    from scipy.signal import lfilter

    if len(data) == 0:
        return -np.inf
    for b, a in _k_weighting_coefficients(fs):
        data = lfilter(b, a, data, axis=0)
    energy = (data**2).sum(axis=1)  # channel-summed, per sample
    block, hop = int(round(BLOCK_S * fs)), int(round(BLOCK_HOP_S * fs))
    if len(energy) < block:
        block, hop = len(energy), len(energy)
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    starts = np.arange(0, len(energy) - block + 1, hop)
    block_ms = (cumulative[starts + block] - cumulative[starts]) / block
    block_lufs = np.array([_lufs(ms) for ms in block_ms])

    kept = block_ms[block_lufs > ABSOLUTE_GATE_LUFS]
    if len(kept) == 0:
        return -np.inf
    relative_gate = _lufs(kept.mean()) + RELATIVE_GATE_LU
    kept = block_ms[(block_lufs > ABSOLUTE_GATE_LUFS) & (block_lufs > relative_gate)]
    return float(_lufs(kept.mean())) if len(kept) else -np.inf


def loudness(audio: Union[str, Path, Audio]) -> float:
    """Integrated loudness of ``audio`` in LUFS (ITU-R BS.1770-4, gated).

    Returns ``-inf`` for silence (nothing above the -70 LUFS absolute gate).
    Every channel is weighted 1 (right for mono and stereo). A clip shorter
    than one 400 ms block is measured as a single block.
    """
    audio = audio if isinstance(audio, Audio) else Audio(audio)
    return _integrated_lufs(*_samples(audio))


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
    return _noise_floor(*_samples(audio), frame_s=frame_s, percentile=percentile)


def apply_gain(audio: Audio, db: float) -> Audio:
    """``audio`` with ``db`` decibels of gain, computed in float.

    The result is converted back to ``audio``'s integer sample width, so peaks
    pushed beyond full scale saturate; :func:`level_turns` keeps its gains in
    float and only converts after its limiter.
    """
    data, _ = _samples(audio)
    gained = data * 10.0 ** (db / 20.0)
    return _to_audio(gained, audio._get_segment(), time_unit=audio.time_unit)


# --- ffmpeg filters -----------------------------------------------------------


def _ffmpeg_float(data: np.ndarray, fs: int, filters: str) -> np.ndarray:
    """``data`` (float ``(n, channels)``) through an ffmpeg ``-af`` chain, unclipped."""
    ch = data.shape[1]
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", _FLOAT_PCM, "-ar", str(fs), "-ac", str(ch),
         "-i", "pipe:0", "-af", filters, "-f", _FLOAT_PCM, "-ar", str(fs),
         "-ac", str(ch), "pipe:1"],
        input=np.ascontiguousarray(data, dtype="<f8").tobytes(),
        capture_output=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg filter {filters!r} failed: {proc.stderr.decode(errors='replace')}"
        )
    out = np.frombuffer(proc.stdout, dtype="<f8")
    return out[: len(out) // ch * ch].reshape(-1, ch)


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
    out = _ffmpeg_float(np.vstack([probe, np.zeros((max_lag, channels))]), fs, filters)
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
    out = _ffmpeg_float(np.vstack([data, np.zeros((lag, ch))]), fs, filters)[lag:]
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
    data, fs = _samples(audio)
    out = _run_filter(
        data,
        fs,
        filters,
        compensate_latency=compensate_latency,
        max_latency_s=max_latency_s,
    )
    return _to_audio(out, audio._get_segment(), time_unit=audio.time_unit)


def _as_stage(
    spec: AudioFilter | None, fs: int, like, *, compensate_latency: bool = False
) -> FloatStage | None:
    """A float stage from an ffmpeg filter string or an ``Audio -> Audio`` callable."""
    if spec is None:
        return None
    if callable(spec):

        def via_audio(x: np.ndarray) -> np.ndarray:
            out = spec(_to_audio(x, like))  # integer PCM: saturates above 0 dBFS
            return _segment_samples(conform_segment(out._get_segment(), like))

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
        data, fs = _segment_samples(like), like.frame_rate
        a, b = int(round(core[0] * fs)), int(round(core[1] * fs))

        def measure(x: np.ndarray) -> float:
            return _integrated_lufs(x[a:b], fs)

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
        return _to_audio(data, like)

    result = process_turns(audio, turns, process, pad_s=pad_s)
    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_leveled.wav",
    )
