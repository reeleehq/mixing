"""Loudness measurement and per-turn loudness matching across speakers.

``loudness(audio)`` is an ITU-R BS.1770-4 / EBU R128 *integrated* loudness
meter (K-weighting, 400 ms blocks, absolute and relative gating) in pure
numpy/scipy, so measuring needs no ffmpeg. ``level_turns`` uses it to bring
every speaker turn of a recording to the same loudness, which whole-track peak
normalisation (:meth:`Audio.normalize`) cannot do.

Per turn, ``level_turns`` runs: the speaker's EQ chain (on the turn padded so
filters settle) -> pre-gain to a common level -> compressor -> gain to
``target_lufs`` -> optional denoise when the noise floor is high -> short fades.
The result is limited at ``ceiling_db``. Filters are ffmpeg filter strings (the
``chains``/``compressor``/``denoise`` seams also accept any ``Audio -> Audio``
callable).

>>> import numpy as np
>>> _k_weighting_coefficients(48000)[0][0] > 0
True
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import ffmpeg_exe, require_package
from .audio_ops import Audio, _frame_rms_db

Turn = tuple[str, float, float]  # (speaker, start_s, end_s)
AudioFilter = Union[str, Callable[[Audio], Audio]]  # ffmpeg filter string or callable

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
DEFAULT_PAD_S = 0.5
DEFAULT_FADE_S = 0.02
DEFAULT_CEILING_DB = -1.5
NOISE_FRAME_S = 0.05
NOISE_PERCENTILE = 10


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
    Every channel is weighted 1 (right for mono and stereo).
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


def apply_ffmpeg_filter(audio: Audio, filters: str) -> Audio:
    """Run an ffmpeg ``-af`` filter chain over ``audio``; returns a new Audio.

    Keeps the sample rate and channel count; output is 16-bit PCM.
    """
    segment = audio._get_segment()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, "in.wav"), Path(tmp, "out.wav")
        segment.export(str(src), format="wav")
        proc = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-af", filters,
             "-ar", str(segment.frame_rate), "-ac", str(segment.channels),
             "-c:a", "pcm_s16le", str(dst)],
            capture_output=True,
        )  # fmt: skip
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg filter {filters!r} failed: {proc.stderr.decode(errors='replace')}"
            )
        result = require_package("pydub").AudioSegment.from_file(str(dst))
    return Audio(result, time_unit=audio.time_unit)


def _as_filter(spec: AudioFilter) -> Callable[[Audio], Audio]:
    if callable(spec):
        return spec
    return lambda audio: apply_ffmpeg_filter(audio, spec)


def _gain_to(audio: Audio, target_lufs: float) -> Audio:
    """``audio`` gained to ``target_lufs``; unchanged if it is silent."""
    current = loudness(audio)
    return (
        audio if not np.isfinite(current) else apply_gain(audio, target_lufs - current)
    )


def _check_turns(turns: Sequence[Turn], duration: float) -> list[Turn]:
    ordered = sorted(turns, key=lambda t: t[1])
    previous_end = 0.0
    for speaker, start, end in ordered:
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError(
                f"bad turn {(speaker, start, end)} for {duration:.2f}s audio"
            )
        if start < previous_end - 1e-6:
            raise ValueError(f"turn {(speaker, start, end)} overlaps the previous one")
        previous_end = end
    return ordered


def level_turns(
    audio: Union[str, Path, Audio],
    turns: Sequence[Turn],
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    chains: Mapping[str, AudioFilter] | None = None,
    pregain_lufs: float = DEFAULT_PREGAIN_LUFS,
    compressor: AudioFilter | None = DEFAULT_COMPRESSOR,
    denoise: AudioFilter = DEFAULT_DENOISE,
    denoise_above: float = DEFAULT_DENOISE_ABOVE_DBFS,
    pad_s: float = DEFAULT_PAD_S,
    fade_s: float = DEFAULT_FADE_S,
    ceiling_db: float | None = DEFAULT_CEILING_DB,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Bring every speaker turn of ``audio`` to the same loudness.

    Args:
        audio: the recording (path or :class:`Audio`).
        turns: ``[(speaker, start_s, end_s), ...]``; every boundary should lie
            in a pause. Audio between turns is kept untouched.
        target_lufs: integrated loudness every turn ends at.
        chains: ``{speaker: ffmpeg filter string | Audio -> Audio}`` per-speaker
            EQ, run on the turn padded by ``pad_s`` so filters settle, then
            trimmed. Speakers without an entry get no EQ.
        pregain_lufs: level each turn is set to before the compressor, so it
            treats all voices alike.
        compressor: ffmpeg filter / callable / ``None`` to skip.
        denoise: filter applied to a turn whose noise floor (10th percentile of
            50 ms RMS), after levelling, is above ``denoise_above`` dBFS.
        fade_s: fade at each turn boundary, hiding the seam.
        ceiling_db: master peak limit in dBFS; ``None`` to skip.
        output: see :mod:`mixing.egress` (``None`` returns the ``Audio``).

    Returns:
        The levelled :class:`Audio` (or the saved ``Path``).
    """
    audio = audio if isinstance(audio, Audio) else Audio(audio)
    duration = audio.duration
    ordered = _check_turns(turns, duration)
    chains = chains or {}
    compress = _as_filter(compressor) if compressor else None
    clean = _as_filter(denoise)
    fade_ms = int(round(fade_s * 1000))

    def level(speaker: str, start: float, end: float) -> Audio:
        lo, hi = max(0.0, start - pad_s), min(duration, end + pad_s)
        padded = audio[lo:hi]
        if speaker in chains:
            padded = _as_filter(chains[speaker])(padded)
        turn = padded[start - lo : start - lo + (end - start)]
        turn = _gain_to(turn, pregain_lufs)
        if compress:
            turn = compress(turn)
        turn = _gain_to(turn, target_lufs)
        if noise_floor_db(turn) > denoise_above:
            turn = _gain_to(clean(turn), target_lufs)
        return Audio(
            turn._get_segment().fade_in(fade_ms).fade_out(fade_ms),
            time_unit=audio.time_unit,
        )

    pieces: list[Audio] = []
    cursor = 0.0
    for speaker, start, end in ordered:
        if start > cursor:
            pieces.append(audio[cursor:start])
        pieces.append(level(speaker, start, end))
        cursor = end
    if cursor < duration:
        pieces.append(audio[cursor:duration])

    result = pieces[0]
    for piece in pieces[1:]:
        result = result + piece
    if ceiling_db is not None:
        limit = 10.0 ** (ceiling_db / 20.0)
        result = apply_ffmpeg_filter(result, f"alimiter=limit={limit:.4f}")

    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_leveled.wav",
    )
