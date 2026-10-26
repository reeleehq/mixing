"""Array-only loudness: meters, group leveling and a peak limiter.

Everything here takes plain float arrays shaped ``(n, channels)`` in [-1, 1]
plus a sample rate, and needs only numpy and scipy (extra ``loudness``,
``pip install 'mixing[loudness]'``), so a package that holds samples (not an
:class:`~mixing.audio.Audio`) can measure and level without the media stack.
:mod:`mixing.audio.leveling` re-exports these and adds the ``Audio`` layer.

- :func:`integrated_lufs`: BS.1770-4 gated integrated loudness. ``meter`` picks
  the strategy: ``"numpy"`` (default, no ffmpeg), ``"ffmpeg_ebur128"``,
  ``"pyloudnorm"``, or any ``(data, fs) -> lufs`` callable.
- :func:`level_group`: the one gain that brings a group of segments (a voice's
  lines) to a target, measured back to back so the group keeps its dynamics.
- :func:`limit_peaks`: a lookahead peak limiter; sample peak by default,
  ``true_peak=True`` holds inter-sample peaks too.
- :func:`true_peak_db`, :func:`loudness_range_lu`: the dBTP and LRA meters.

>>> import numpy as np
>>> integrated_lufs(np.zeros((100, 1)), 8000)
-inf
>>> limit_peaks(np.array([[2.0], [0.5]]), ceiling=1.0)[:, 0].tolist()
[1.0, 0.25]
"""

from __future__ import annotations

import re
import subprocess
import warnings
from typing import Callable, Optional, Sequence, Union

try:
    import numpy as np
except ImportError as e:  # numpy is the extra's own requirement
    raise ImportError(
        "mixing.loudness needs numpy and scipy: pip install 'mixing[loudness]'"
    ) from e

from mixing.util import ffmpeg_exe

#: A loudness meter: float samples ``(n, channels)`` and a rate in, LUFS out.
Meter = Union[str, Callable[[np.ndarray, int], float]]

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


#: Group leveling defaults.
#: Most a group may be raised (dB). A group needing more is mostly silence or
#: room tone; boosting it further only amplifies noise, so the gain is capped
#: and a warning names it.
MAX_GAIN_DB = 24.0
#: A group shorter than this is measured looped to this length: the gate works
#: in 400 ms blocks, and a group of short lines would read as silence.
MIN_MEASURE_S = 1.0
#: The limiter's default lookahead / release (s): the gain eases into a peak
#: and back out over this much on each side, so nothing below ~20 Hz is
#: modulated within a cycle.
DEFAULT_LIMITER_WINDOW_S = 0.05
_EBUR128_I = re.compile(r"I:\s*(-?\d+(?:\.\d+)?|-inf)\s*LUFS")
_FFMPEG_FLOAT = "f32le"


# --- BS.1770 meter (numpy) ---------------------------------------------------


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


def numpy_lufs(data: np.ndarray, fs: int, *, chunk_s: float = FILTER_CHUNK_S) -> float:
    """BS.1770-4 gated integrated loudness of float samples ``(n, channels)``, in numpy."""
    if len(data) == 0:
        return -np.inf
    energy = _k_weighted_energy(data, fs, chunk_s=chunk_s)
    block_ms = _block_mean_squares(energy, fs, block_s=BLOCK_S, hop_s=BLOCK_HOP_S)
    kept, _ = _gated(block_ms, RELATIVE_GATE_LU)
    return float(_lufs(kept.mean())) if len(kept) else -np.inf


def loudness_range_lu(data: np.ndarray, fs: int) -> float:
    """Loudness range (LRA) of float samples in LU (EBU Tech 3342).

    The spread between the 10th and 95th percentile of the gated 3 s
    short-term loudness. 0.0 for silence or a signal shorter than 3 s.
    """
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


# --- other meters ----------------------------------------------------------------


def ebur128_lufs(data: np.ndarray, fs: int) -> float:
    """Integrated loudness by ffmpeg's ``ebur128`` filter (``-inf`` for silence)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostats", "-nostdin", "-f", _FFMPEG_FLOAT,
         "-ar", str(fs), "-ac", str(data.shape[1]), "-i", "pipe:0",
         "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        input=np.ascontiguousarray(data, dtype="<f4").tobytes(),
        capture_output=True,
    )  # fmt: skip
    stderr = proc.stderr.decode(errors="replace")
    found = _EBUR128_I.findall(stderr)
    if proc.returncode != 0 or not found:
        raise RuntimeError(
            f"ffmpeg could not measure loudness (rc={proc.returncode}): "
            f"{stderr.strip()[-400:]}"
        )
    return float("-inf") if found[-1] == "-inf" else float(found[-1])


def pyloudnorm_lufs(data: np.ndarray, fs: int) -> float:
    """Integrated loudness by ``pyloudnorm`` (needs it installed; >= 0.4 s of audio)."""
    from mixing.util import require_package

    meter = require_package("pyloudnorm").Meter(fs)
    return float(meter.integrated_loudness(data))


#: The named meters of :func:`integrated_lufs`.
METERS: dict[str, Callable[[np.ndarray, int], float]] = {
    "numpy": numpy_lufs,
    "ffmpeg_ebur128": ebur128_lufs,
    "pyloudnorm": pyloudnorm_lufs,
}


def integrated_lufs(data: np.ndarray, fs: int, *, meter: Meter = "numpy") -> float:
    """Integrated loudness (LUFS) of float samples ``(n, channels)`` at ``fs``.

    ``meter`` is a name from :data:`METERS` or any ``(data, fs) -> lufs``
    callable. ``-inf`` for silence.
    """
    if callable(meter):
        return float(meter(data, fs))
    try:
        return float(METERS[meter](data, fs))
    except KeyError:
        raise ValueError(f"unknown meter {meter!r}; one of {sorted(METERS)}") from None


# --- true peak -----------------------------------------------------------------


def true_peak_envelope(
    data: np.ndarray, fs: int, *, oversampling: Optional[int] = None
) -> np.ndarray:
    """Per-sample true peak ``(n,)``: the largest ``|x|`` over all channels of the
    oversampled signal in ``[n, n+1)``.

    ``oversampling`` defaults to 4x, raised for rates under 48 kHz so the
    oversampled rate is at least ``TRUE_PEAK_MIN_RATE``. Chunked, with context
    shared between chunks so the upsampling filter does not ring at the joins.
    """
    from scipy.signal import resample_poly

    factor = oversampling or max(TRUE_PEAK_OVERSAMPLING, -(-TRUE_PEAK_MIN_RATE // fs))
    step = max(1, int(round(FILTER_CHUNK_S * fs)))
    margin = int(round(TRUE_PEAK_MARGIN_S * fs))
    envelope = np.empty(len(data))
    for start in range(0, len(data), step):
        stop = min(start + step, len(data))
        lo, hi = max(0, start - margin), min(len(data), stop + margin)
        up = np.abs(resample_poly(data[lo:hi], factor, 1, axis=0)).max(axis=1)
        inner = up[(start - lo) * factor : (stop - lo) * factor]
        envelope[start:stop] = inner.reshape(-1, factor).max(axis=1)
    return envelope


def true_peak_db(
    data: np.ndarray, fs: int, *, oversampling: Optional[int] = None
) -> float:
    """True peak of float samples in dBTP (``-inf`` for silence)."""
    if len(data) == 0:
        return -np.inf
    peak = float(true_peak_envelope(data, fs, oversampling=oversampling).max())
    return float(20.0 * np.log10(peak)) if peak > 0 else -np.inf


# --- group leveling -------------------------------------------------------------


def _join(segments: Sequence[np.ndarray], fs: int, min_measure_s: float) -> np.ndarray:
    """Segments back to back, keeping their channels when all have the same
    count (BS.1770 weights channels; a downmix of uncorrelated stereo reads
    ~3 dB low), else downmixed to mono; looped up to ``min_measure_s``."""
    segments = [np.asarray(s, dtype=np.float64) for s in segments]
    if not segments:
        return np.zeros((0, 1))
    keep = len({s.shape[1] for s in segments}) == 1
    joined = np.concatenate(
        [s if keep else s.mean(axis=1, keepdims=True) for s in segments]
    )
    need = int(min_measure_s * fs)
    if 0 < len(joined) < need:
        joined = np.tile(joined, (-(-need // len(joined)), 1))
    return joined


def level_group(
    segments: Sequence[np.ndarray],
    fs: int,
    *,
    target_lufs: float,
    meter: Meter = "numpy",
    max_gain_db: float = MAX_GAIN_DB,
    min_measure_s: float = MIN_MEASURE_S,
) -> tuple[float, float]:
    """``(gain_db, measured_lufs)``: the one gain bringing a group to ``target_lufs``.

    The group (a voice's lines, say) is measured **back to back**, so its
    members keep their relative dynamics instead of each being pulled to the
    target. A group shorter than ``min_measure_s`` is looped to that length for
    the measurement. Silence (``measured_lufs == -inf``) gets gain 0. A gain
    above ``max_gain_db`` is capped, with a warning, since it would mostly
    amplify noise. Apply it with ``data * 10 ** (gain_db / 20)`` and
    :func:`limit_peaks`.
    """
    measured = integrated_lufs(_join(segments, fs, min_measure_s), fs, meter=meter)
    if not np.isfinite(measured):
        return 0.0, measured
    gain = target_lufs - measured
    if gain > max_gain_db:
        warnings.warn(
            f"group measures {measured:.1f} LUFS: reaching {target_lufs:g} would take "
            f"{gain:.1f} dB, mostly noise; capped at +{max_gain_db:g} dB",
            stacklevel=2,
        )
        gain = max_gain_db
    return float(gain), measured


# --- limiter --------------------------------------------------------------------


def _sliding_min(x: np.ndarray, k: int) -> np.ndarray:
    """The minimum of every ``k``-long window of ``x`` (``len(x) - k + 1`` of
    them), in O(n) (van Herk / Gil-Werman: prefix and suffix minima per block)."""
    n = len(x)
    pad = (-n) % k
    y = np.concatenate([x, np.full(pad, np.inf)]).reshape(-1, k)
    prefix = np.minimum.accumulate(y, axis=1).ravel()
    suffix = np.minimum.accumulate(y[:, ::-1], axis=1)[:, ::-1].ravel()
    i = np.arange(n - k + 1)
    return np.minimum(suffix[i], prefix[i + k - 1])


def limit_peaks(
    data: np.ndarray,
    *,
    ceiling: float,
    window: int = 1,
    true_peak: bool = False,
    fs: Optional[int] = None,
) -> np.ndarray:
    """``data`` ``(n, channels)`` with no peak above ``ceiling`` (linear, full scale 1.0).

    The gain each sample needs (``ceiling / peak``, at most 1) is spread over
    ``window`` samples on both sides -- a sliding minimum over ``2 * window +
    1``, then a moving average over ``window`` -- so it eases into and out of
    a peak. Every sample's gain is at most what that sample needs, so the
    ceiling holds exactly; nothing moves in time. ``window`` is in samples
    (``DEFAULT_LIMITER_WINDOW_S`` * ``fs`` is a good size).

    With ``true_peak=True`` (needs ``fs``) the need comes from the oversampled
    peak (see :func:`true_peak_envelope`) of each sample and the gap before it,
    so inter-sample peaks are held at ``ceiling`` too, up to the small error of
    interpolating the gain between samples.
    """
    if not len(data):
        return data
    if true_peak:
        if fs is None:
            raise ValueError("true_peak=True needs fs")
        peak = true_peak_envelope(data, fs)
        peak = np.maximum(peak, np.concatenate([[0.0], peak[:-1]]))  # the gap before
    else:
        peak = np.abs(data).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if need.min() >= 1.0:
        return data
    w = max(1, int(window))
    lows = _sliding_min(np.pad(need, (w, w), constant_values=1.0), 2 * w + 1)
    padded = np.pad(lows, (w // 2, w - 1 - w // 2), mode="edge")
    gain = np.convolve(padded, np.ones(w) / w, mode="valid")
    gain = np.minimum(gain, need)  # float rounding never lets a peak through
    return data * gain[:, None]
