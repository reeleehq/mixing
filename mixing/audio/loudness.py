"""Loudness measurement and per-turn loudness matching across speakers.

``loudness(audio)`` is an ITU-R BS.1770-4 / EBU R128 *integrated* loudness
meter (K-weighting, 400 ms blocks, absolute and relative gating) in pure
numpy/scipy, so measuring needs no ffmpeg. ``level_turns`` uses it to bring
every speaker turn of a recording to the same loudness, which whole-track peak
normalisation (:meth:`Audio.normalize`) cannot do.

``level_turns`` is :func:`mixing.audio.turns.process_turns` with a chain of
``Audio -> Audio`` stages per turn, all run on the turn padded by ``pad_s``
(so filters settle, and filter latency is compensated before the trim): the
speaker's EQ -> pre-gain to a common level -> compressor -> gain to
``target_lufs`` -> denoise when the noise floor is high -> short fades. The
result is peak-limited at ``ceiling_db``. Filters are ffmpeg filter strings
(the ``chains``/``compressor``/``denoise`` seams also accept any
``Audio -> Audio`` callable).

>>> from pydub import AudioSegment
>>> loudness(Audio(AudioSegment.silent(1000)))
-inf
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe
from .audio_ops import Audio, _frame_rms_db
from .turns import DEFAULT_PAD_S, Core, Turn, check_turns, process_turns

AudioFilter = Union[str, Callable[[Audio], Audio]]  # ffmpeg filter string or callable
Stage = Optional[Callable[[Audio], Audio]]

# Kept for callers/tests that used the private name before ``turns`` existed.
_check_turns = check_turns

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
#: Most a turn's loudness may be raised (dB). A turn needing more is mostly
#: silence or room tone, and boosting it further only amplifies noise; it is
#: capped here and a warning names it.
MAX_GAIN_DB = 24.0
NOISE_FRAME_S = 0.05
NOISE_PERCENTILE = 10
#: Longest filter latency (s) ``apply_ffmpeg_filter`` searches for and removes
#: (``afftdn`` delays by ~25 ms).
MAX_FILTER_LATENCY_S = 0.1
#: Length (s) of the loudest excerpt the latency search correlates over.
LATENCY_PROBE_S = 2.0

#: ffmpeg raw PCM format for each pydub sample width (bytes).
_PCM_FORMATS = {1: "u8", 2: "s16le", 3: "s24le", 4: "s32le"}


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


def _samples(audio: Audio) -> tuple[np.ndarray, int]:
    """``(float samples shaped (n, channels) in [-1, 1], rate)`` of ``audio``."""
    segment = audio._get_segment()
    data = np.array(segment.get_array_of_samples(), dtype=np.float64)
    data = data.reshape(-1, segment.channels) / float(
        1 << (8 * segment.sample_width - 1)
    )
    return data, segment.frame_rate


def _lufs(mean_square: float) -> float:
    return _LUFS_OFFSET + 10.0 * np.log10(mean_square) if mean_square > 0 else -np.inf


def loudness(audio: Union[str, Path, Audio]) -> float:
    """Integrated loudness of ``audio`` in LUFS (ITU-R BS.1770-4, gated).

    Returns ``-inf`` for silence (nothing above the -70 LUFS absolute gate).
    Every channel is weighted 1 (right for mono and stereo). A clip shorter
    than one 400 ms block is measured as a single block.
    """
    from scipy.signal import lfilter

    audio = audio if isinstance(audio, Audio) else Audio(audio)
    data, fs = _samples(audio)
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


def noise_floor_db(
    audio: Audio,
    *,
    frame_s: float = NOISE_FRAME_S,
    percentile: float = NOISE_PERCENTILE,
) -> float:
    """Noise floor estimate: a low percentile of the ``frame_s`` RMS levels (dBFS)."""
    segment = audio._get_segment()
    frame_n = max(1, int(round(frame_s * segment.frame_rate)))
    return float(np.percentile(_frame_rms_db(segment, frame_n=frame_n), percentile))


def apply_gain(audio: Audio, db: float) -> Audio:
    """``audio`` with ``db`` decibels of gain applied."""
    return Audio(audio._get_segment().apply_gain(db), time_unit=audio.time_unit)


# --- ffmpeg filters -----------------------------------------------------------


def _filter_latency(
    source: np.ndarray, result: np.ndarray, *, max_lag: int, probe_n: int
) -> int:
    """Delay (samples, ``0..max_lag``) of ``result`` relative to ``source`` (mono).

    Cross-correlates the ``probe_n`` samples of ``source`` around its peak
    against ``result``, so the cost does not grow with the clip length.
    """
    if max_lag <= 0 or not np.any(source):
        return 0
    probe_n = min(len(source), max(probe_n, 4 * max_lag))
    centre = int(np.argmax(np.abs(source)))
    lo = max(0, min(centre - probe_n // 2, len(source) - probe_n))
    ref = source[lo : lo + probe_n]
    seg = result[lo : lo + probe_n + max_lag]
    if len(seg) < len(ref):
        return 0
    n = 1 << int(np.ceil(np.log2(len(seg) + len(ref))))
    xcorr = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(ref, n)), n)
    return int(np.argmax(xcorr[: max_lag + 1]))


def apply_ffmpeg_filter(
    audio: Audio,
    filters: str,
    *,
    compensate_latency: bool = True,
    max_latency_s: float = MAX_FILTER_LATENCY_S,
) -> Audio:
    """Run an ffmpeg ``-af`` filter chain over ``audio``; returns a new Audio.

    Keeps the sample rate, channel count, sample width and length. Audio is
    piped as raw PCM (no temp files). With ``compensate_latency`` the input is
    padded by ``max_latency_s`` of silence, the filter's delay (e.g. ``afftdn``'s
    ~25 ms) is found by cross-correlation and removed, so onsets stay put.
    """
    segment = audio._get_segment()
    fs, ch, width = segment.frame_rate, segment.channels, segment.sample_width
    fmt = _PCM_FORMATS[width]
    max_lag = int(round(max_latency_s * fs)) if compensate_latency else 0
    frame_bytes = segment.frame_width
    n_frames = len(segment.raw_data) // frame_bytes
    raw_in = segment.raw_data + _silence_bytes(max_lag * ch, width)
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-f", fmt, "-ar", str(fs), "-ac", str(ch),
         "-i", "pipe:0", "-af", filters, "-f", fmt, "-ar", str(fs), "-ac", str(ch),
         "pipe:1"],
        input=raw_in,
        capture_output=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg filter {filters!r} failed: {proc.stderr.decode(errors='replace')}"
        )
    out = proc.stdout
    lag = 0
    if max_lag:
        lag = _filter_latency(
            _mono(segment.raw_data, width, ch),
            _mono(out[: len(out) // frame_bytes * frame_bytes], width, ch),
            max_lag=max_lag,
            probe_n=int(round(LATENCY_PROBE_S * fs)),
        )
    out = out[lag * frame_bytes : (lag + n_frames) * frame_bytes]
    out += _silence_bytes((n_frames * frame_bytes - len(out)) // width, width)
    return Audio(segment._spawn(out), time_unit=audio.time_unit)


def _silence_bytes(n_samples: int, width: int) -> bytes:
    return (b"\x80" if width == 1 else b"\0" * width) * max(0, n_samples)


def _mono(raw: bytes, width: int, channels: int) -> np.ndarray:
    """Mono float mixdown of raw PCM (any width ffmpeg/pydub use)."""
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int8).astype(np.int32) << 16))  # fmt: skip
    else:
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if width == 1:
            data = data - 128
    data = np.asarray(data, dtype=np.float64)
    return data.reshape(-1, channels).mean(axis=1)


def _as_filter(spec: AudioFilter | None) -> Stage:
    if spec is None or callable(spec):
        return spec
    return lambda audio: apply_ffmpeg_filter(audio, spec)


# --- level_turns stages -------------------------------------------------------


def _frames(audio: Audio, core: Core) -> Audio:
    """The ``core`` ``(start_s, end_s)`` part of ``audio``, cut on sample frames."""
    segment = audio._get_segment()
    fs, width = segment.frame_rate, segment.frame_width
    a, b = int(round(core[0] * fs)), int(round(core[1] * fs))
    return Audio(segment._spawn(segment.raw_data[a * width : b * width]))


def _gain_to(audio: Audio, target_lufs: float, *, core: Core) -> Audio:
    """``audio`` gained so its ``core`` measures ``target_lufs``; silence unchanged."""
    current = loudness(_frames(audio, core))
    return (
        audio if not np.isfinite(current) else apply_gain(audio, target_lufs - current)
    )


def _fade(audio: Audio, core: Core, fade_s: float) -> Audio:
    """Linear fade in at ``core[0]`` and out at ``core[1]`` (capped to half the turn)."""
    segment = audio._get_segment()
    fs, ch = segment.frame_rate, segment.channels
    samples = np.array(segment.get_array_of_samples()).reshape(-1, ch)
    b = min(len(samples), int(round(core[1] * fs)))  # a stage may have shortened it
    a = min(b, int(round(core[0] * fs)))
    n = min(int(round(fade_s * fs)), (b - a) // 2)
    if n <= 0 or segment.sample_width == 1:
        return audio
    ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    scaled = samples.astype(np.float64)
    scaled[a : a + n] *= ramp
    scaled[b - n : b] *= ramp[::-1]
    out = np.round(scaled).astype(samples.dtype)
    return Audio(segment._spawn(out.tobytes()), time_unit=audio.time_unit)


def _chain(stages: Sequence[Stage]) -> Callable[[Audio], Audio]:
    """Compose ``Audio -> Audio`` stages left to right; ``None`` stages are skipped."""

    def run(audio: Audio) -> Audio:
        for stage in stages:
            if stage is not None:
                audio = stage(audio)
        return audio

    return run


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
            ``None`` to skip.
        pad_s: context kept around each turn while filtering.
        fade_s: fade at each turn boundary, hiding the seam.
        max_gain_db: most a turn's loudness may be raised; beyond it the gain
            is capped and a warning is issued.
        ceiling_db: master peak limit in dBFS (a true limiter, no auto-level,
            latency-compensated); ``None`` to skip.
        output: see :mod:`mixing.egress` (``None`` returns the ``Audio``).

    Notes:
        - Audio **between** turns is left untouched. Room tone *inside* a turn
          is gained with the voice, so the noise floor can step at turn
          boundaries.
        - A turn shorter than 400 ms is measured as one block (BS.1770 needs
          at least one full block for gating).

    Returns:
        The levelled :class:`Audio` (or the saved ``Path``).
    """
    chains = chains or {}
    compress = _as_filter(compressor)
    clean = _as_filter(denoise)

    def process(speaker: str, padded: Audio, core: Core) -> Audio:
        def gain_to(target: float) -> Stage:
            return lambda x: _gain_to(x, target, core=core)

        def denoise_if_noisy(x: Audio) -> Audio:
            if noise_floor_db(_frames(x, core)) <= denoise_above:
                return x
            return _gain_to(clean(x), target, core=core)

        padded = _chain([_as_filter(chains.get(speaker))])(padded)
        original = loudness(_frames(padded, core))
        target, pregain = target_lufs, pregain_lufs
        if np.isfinite(original) and target - original > max_gain_db:
            warnings.warn(
                f"turn {speaker!r} at {original:.1f} LUFS needs "
                f"{target - original:.1f} dB to reach {target} LUFS; "
                f"gain capped at {max_gain_db} dB",
                stacklevel=2,
            )
            target = original + max_gain_db
            pregain = min(pregain, target)
        return _chain(
            [
                gain_to(pregain),
                compress,
                gain_to(target),
                denoise_if_noisy if clean else None,
                lambda x: _fade(x, core, fade_s),
            ]
        )(padded)

    result = process_turns(audio, turns, process, pad_s=pad_s)
    if ceiling_db is not None:
        limit = 10.0 ** (ceiling_db / 20.0)
        # level=0: no auto make-up gain (the default raises output to 0 dBFS);
        # latency=1: alimiter compensates its own lookahead delay.
        result = apply_ffmpeg_filter(
            result,
            f"alimiter=limit={limit:.6f}:level=0:latency=1",
            compensate_latency=False,
        )

    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_leveled.wav",
    )
