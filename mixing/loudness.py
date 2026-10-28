"""Array-only loudness: meters, group leveling and a peak limiter.

Everything here takes plain float arrays shaped ``(n, channels)`` in [-1, 1]
(or a stream of them) plus a sample rate, :func:`file_loudness` excepted, and needs only numpy and scipy (extra ``loudness``,
``pip install 'mixing[loudness]'``), so a package that holds samples (not an
:class:`~mixing.audio.Audio`) can measure and level without the media stack.
:mod:`mixing.audio.leveling` builds its ``Audio`` layer on these.

- :func:`integrated_lufs`: BS.1770-4 gated integrated loudness. ``meter`` picks
  the strategy: ``"numpy"`` (default, no ffmpeg), ``"ffmpeg_ebur128"``,
  ``"pyloudnorm"``, or any ``(data, fs) -> lufs`` callable.
- :func:`level_group`: the one gain that brings a group of segments (a voice's
  lines) to a target, measured back to back so the group keeps its dynamics.
- :func:`limit_peaks`: a lookahead peak limiter; sample peak by default,
  ``true_peak=True`` holds inter-sample peaks too.
- :func:`true_peak_db`, :func:`loudness_range_lu`: the dBTP and LRA meters.
- :func:`stream_loudness`, :func:`file_loudness`: integrated loudness and LRA
  in one pass over chunks (or a file decoded in chunks by
  :mod:`mixing.decode`), in memory bounded by the chunk, not the track.
- :func:`stream_limit_peaks`, :func:`level_file`: the limiter over chunks, and
  a file brought to a loudness target with one measured gain and that limiter,
  file in and file out, in the same bounded memory.

>>> import numpy as np
>>> integrated_lufs(np.zeros((100, 1)), 8000)
-inf
>>> limit_peaks(np.array([[2.0], [0.5]]), ceiling=1.0)[:, 0].tolist()
[1.0, 0.25]
"""

from __future__ import annotations

import numbers
import os
import re
import subprocess
import warnings
from contextlib import closing
from functools import lru_cache, partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

from mixing.util import LOUDNESS_EXTRA, ffmpeg_exe, install_hint, require_package

try:  # numpy only: scipy is imported where used, so the ffmpeg meter works without it
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.loudness needs numpy and scipy: {install_hint(LOUDNESS_EXTRA)}",
        name=e.name,
    ) from e

from mixing.decode import (  # noqa: E402
    DECODE_CHUNK_S,
    decode_float_chunks,
    encode_float_chunks,
    stream_kinds,
)
from mixing.egress import Output, write_egress  # noqa: E402

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
#: The oversampling filter: a Kaiser-windowed sinc this many input samples long
#: on each side. Its worst polyphase branch is within 0.002 dB to 0.45 fs and
#: 0.2 dB at 0.47 fs, with images above 0.55 fs 80 dB down (scipy's default,
#: 10 samples and beta 5, loses 1.4 dB at 0.45 fs and 4.5 dB at 0.47 fs). It is
#: also the context neighbouring chunks share, so chunk joins are exact.
TRUE_PEAK_FILTER_HALF_SAMPLES = 32
TRUE_PEAK_FILTER_KAISER_BETA = 8.0
#: ``method="long"``'s filter: the same windowed sinc, 2048 samples a side, a
#: lower beta trading image rejection for a narrower transition at Nyquist.
#: Passband within +-0.0025 dB to 0.497 fs; images ~57 dB down from 0.5005 fs,
#: 71 dB from 0.503 fs. Against an exact sinc reconstruction, hard-clipped
#: full-band noise reads within ~0.05 dB (the tail it truncates; the short
#: filter: 0.2-1 dB low). It is a windowed sinc, not a brickwall FFT
#: upsampler, because a finite filter keeps the chunk joins exact.
TRUE_PEAK_LONG_HALF_SAMPLES = 2048
TRUE_PEAK_LONG_KAISER_BETA = 5.0


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
#: Frames the limiter works on at a time (~5.5 s at 48 kHz). Memory is the
#: output plus a fixed per-chunk overhead (tens of MB; more for a true peak,
#: whose envelope oversamples the chunk), however long the signal.
LIMITER_CHUNK_FRAMES = 1 << 18
#: :func:`level_file`'s default ceiling (dBTP with ``true_peak``, else dBFS):
#: EBU R 128's maximum true-peak level.
DEFAULT_CEILING_DB = -1.0
#: What :func:`level_file` adds to the source's stem when it names the output.
LEVELED_SUFFIX = "_leveled"
_EBUR128_I = re.compile(r"I:\s*(-?\d+(?:\.\d+)?|-inf)\s*LUFS")
_FFMPEG_FLOAT = "f32le"


def _as_2d(data: np.ndarray) -> np.ndarray:
    """``data`` as ``(n, channels)``: a 1-D array is one channel."""
    data = np.asarray(data)
    return data[:, None] if data.ndim == 1 else data


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


def _k_weighted_energies(chunks: Iterable[np.ndarray], fs: int) -> Iterator[np.ndarray]:
    """Channel-summed squared K-weighted samples, one ``(k,)`` array per chunk.

    The filter state is carried across chunks, so the result is the same as
    filtering the whole signal while holding only a chunk at a time.
    """
    lfilter = require_package("scipy.signal", extra=LOUDNESS_EXTRA).lfilter

    sections, states = _k_weighting_coefficients(fs), None
    for x in chunks:
        x = np.asarray(x)
        if x.dtype.kind in "iub":
            raise ValueError(f"samples must be floats in [-1, 1], got {x.dtype}")
        x = _as_2d(x.astype(np.float64, copy=False))
        if not len(x):  # lfilter returns a garbage state for an empty input
            continue
        if states is None:  # biquads: order 2
            states = [np.zeros((2, x.shape[1])) for _ in sections]
        elif x.shape[1] != states[0].shape[1]:
            raise ValueError(
                f"a chunk has {x.shape[1]} channels, the stream {states[0].shape[1]}"
            )
        for k, (b, a) in enumerate(sections):
            x, states[k] = lfilter(b, a, x, axis=0, zi=states[k])
        yield (x**2).sum(axis=1)


class _Blocks:
    """Mean square of every ``block``-long window at ``hop`` spacing, over
    energy fed in pieces; holds at most a block (plus the hop, if it is longer)
    plus one piece. A signal
    shorter than one block is one block over all of it."""

    def __init__(self, block: int, hop: int):
        self.block, self.hop, self.length = block, hop, 0
        self._pieces: list = []  # energy from self._offset on, not yet joined
        self._pending = self._offset = self._next = 0
        self._out: list = []

    def feed(self, energy: np.ndarray) -> None:
        self.length += len(energy)
        self._pieces.append(energy)
        self._pending += len(energy)
        if self._offset + self._pending < self._next + self.block:
            return  # no window complete yet: no join, no copy
        buffer = np.concatenate(self._pieces)
        end = self._offset + len(buffer)
        starts = np.arange(self._next, end - self.block + 1, self.hop)
        cumulative = np.concatenate([[0.0], np.cumsum(buffer)])
        at = starts - self._offset
        self._out.append((cumulative[at + self.block] - cumulative[at]) / self.block)
        self._next = int(starts[-1]) + self.hop
        drop = min(self._next - self._offset, len(buffer))
        self._pieces, self._offset = [buffer[drop:]], self._offset + drop
        self._pending = len(buffer) - drop

    def result(self) -> np.ndarray:
        if self._out:
            return np.concatenate(self._out)
        if not self.length:
            return np.zeros(0)
        return np.concatenate(self._pieces)[None].mean(axis=1)


class LoudnessReading(NamedTuple):
    """Integrated loudness (LUFS, ``-inf`` for silence) and loudness range (LU)."""

    integrated_lufs: float
    loudness_range_lu: float


def _integrated(block_ms: np.ndarray) -> float:
    kept, _ = _gated(block_ms, RELATIVE_GATE_LU)
    return float(_lufs(kept.mean())) if len(kept) else -np.inf


def _loudness_range(windows: np.ndarray, length: int, fs: int) -> float:
    if length < round(LRA_WINDOW_S * fs):
        return 0.0
    _, lufs = _gated(windows, LRA_RELATIVE_GATE_LU)
    if len(lufs) == 0:
        return 0.0
    low, high = np.percentile(lufs, LRA_PERCENTILES)
    return float(high - low)


def _blocks(seconds: float, hop_s: float, fs: int) -> _Blocks:
    return _Blocks(int(round(seconds * fs)), int(round(hop_s * fs)))


def stream_loudness(chunks: Iterable[np.ndarray], fs: int) -> LoudnessReading:
    """Integrated loudness and LRA of a signal that arrives in chunks.

    ``chunks`` are float arrays ``(k, channels)`` (or 1-D, mono) in [-1, 1],
    one after another; any sizes. One pass, holding one chunk plus the block
    energies (a few floats per second), so a track of any length meters in
    bounded memory. Same results as :func:`numpy_lufs` and
    :func:`loudness_range_lu` on the joined signal.
    """
    gate, short_term = (
        _blocks(BLOCK_S, BLOCK_HOP_S, fs),
        _blocks(LRA_WINDOW_S, LRA_HOP_S, fs),
    )
    for energy in _k_weighted_energies(chunks, fs):
        gate.feed(energy)
        short_term.feed(energy)
    if gate.length == 0:
        return LoudnessReading(-np.inf, 0.0)
    return LoudnessReading(
        _integrated(gate.result()),
        _loudness_range(short_term.result(), short_term.length, fs),
    )


def _array_chunks(data: np.ndarray, fs: int, chunk_s: float) -> Iterator[np.ndarray]:
    step = max(1, int(round(chunk_s * fs)))
    return (data[start : start + step] for start in range(0, len(data), step))


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
    """BS.1770-4 gated integrated loudness of float samples, in numpy.

    Every channel is weighted 1.0 (right for mono and stereo; BS.1770's
    surround weights and LFE exclusion for >2 channels are not applied, so
    the ffmpeg meter reads differently there). Filters ``chunk_s`` at a time.
    """
    data = _as_2d(data)
    gate = _blocks(BLOCK_S, BLOCK_HOP_S, fs)
    for energy in _k_weighted_energies(_array_chunks(data, fs, chunk_s), fs):
        gate.feed(energy)
    return _integrated(gate.result()) if gate.length else -np.inf


def loudness_range_lu(data: np.ndarray, fs: int) -> float:
    """Loudness range (LRA) of float samples in LU (EBU Tech 3342).

    The spread between the 10th and 95th percentile of the gated 3 s
    short-term loudness. 0.0 for silence or a signal shorter than 3 s.
    """
    data = _as_2d(data)
    windows = _blocks(LRA_WINDOW_S, LRA_HOP_S, fs)
    for energy in _k_weighted_energies(_array_chunks(data, fs, FILTER_CHUNK_S), fs):
        windows.feed(energy)
    return _loudness_range(windows.result(), windows.length, fs)


def file_loudness(
    path,
    *,
    chunk_s: float = DECODE_CHUNK_S,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LoudnessReading:
    """Integrated loudness and LRA of a media file's audio, in bounded memory.

    :func:`mixing.decode.decode_float_chunks` (``chunk_s``, ``ffmpeg``,
    ``timeout``: see :mod:`mixing.decode`) into
    :func:`stream_loudness`: a 30 min stereo track meters at ~130 MB peak RSS
    (~106 MB of it numpy and scipy), the same as a 5 min one, in about a
    seventh of ffmpeg ``loudnorm``'s time. Against ``loudnorm``'s
    ``input_i``: within ~0.05 LU at 44.1/48 kHz; ~0.35 LU apart at 8-16 kHz
    (``loudnorm`` upsamples to 192 kHz; ffmpeg's ``ebur128`` agrees with
    this meter); 5.1 reads ~0.1 LU apart (no surround weights, see
    :func:`numpy_lufs`).
    """
    rate, chunks = decode_float_chunks(
        path, chunk_s=chunk_s, ffmpeg=ffmpeg, timeout=timeout
    )
    with closing(chunks):  # stops ffmpeg even if metering raises partway
        return stream_loudness(chunks, rate)


# --- other meters ----------------------------------------------------------------


def ebur128_lufs(data: np.ndarray, fs: int) -> float:
    """Integrated loudness by ffmpeg's ``ebur128`` filter (``-inf`` for silence).

    Needs at least one 400 ms block (shorter reads as silence), and ffmpeg
    assumes a channel layout beyond stereo.
    """
    data = _as_2d(data)
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
    value = float(found[-1])  # ebur128 reports silence as -70, not -inf
    return value if value > ABSOLUTE_GATE_LUFS else float("-inf")


def pyloudnorm_lufs(data: np.ndarray, fs: int) -> float:
    """Integrated loudness by ``pyloudnorm`` (needs it installed; >= 0.4 s of
    audio, at most 5 channels -- it raises otherwise)."""
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
    callable. ``-inf`` for silence. A 1-D array is one channel.
    """
    if callable(meter):
        return float(meter(data, fs))
    try:
        return float(METERS[meter](data, fs))
    except KeyError:
        raise ValueError(f"unknown meter {meter!r}; one of {sorted(METERS)}") from None


# --- true peak -----------------------------------------------------------------


@lru_cache(maxsize=None)
def _interpolation_filter(factor: int, half_samples: int, beta: float) -> np.ndarray:
    """The Kaiser-windowed sinc that upsamples by ``factor``, ``half_samples``
    input samples long on each side, its centre tap 1. Its every ``factor``-th
    tap is zero but the centre one, so the original samples pass unchanged and
    a true peak never reads below the sample peak. Read-only: it is cached."""
    kaiser = require_package("scipy.signal.windows", extra=LOUDNESS_EXTRA).kaiser

    half = half_samples * factor
    n = np.arange(-half, half + 1)
    taps = np.sinc(n / factor) * kaiser(len(n), beta)
    taps[n % factor == 0] = 0.0  # exactly, not sinc's ~1e-17
    taps[half] = 1.0
    taps.flags.writeable = False
    return taps


def _upsample(
    x: np.ndarray, factor: int, *, half_samples: int, beta: float
) -> np.ndarray:
    """``x`` ``(n, channels)`` upsampled by ``factor`` with the windowed sinc
    ``half_samples`` a side (:func:`_interpolation_filter`), polyphase: phase 0
    is ``x`` itself (so the samples pass unchanged, exactly), each in-between
    phase is ``x`` convolved (overlap-add FFT) with every ``factor``-th tap."""
    oaconvolve = require_package("scipy.signal", extra=LOUDNESS_EXTRA).oaconvolve
    taps = _interpolation_filter(factor, half_samples, beta)
    x = np.asarray(x, dtype=np.float64)
    up = np.empty((len(x) * factor, x.shape[1]))
    up[::factor] = x
    for phase in range(1, factor):
        full = oaconvolve(x, taps[phase::factor, None], axes=0)
        up[phase::factor] = full[half_samples : half_samples + len(x)]
    return up


class _Interpolation(NamedTuple):
    """A true-peak interpolation filter: a Kaiser-windowed sinc ``half_samples``
    input samples long on each side, with window ``beta``. ``half_samples`` is
    also the context neighbouring chunks share, so chunk joins are exact."""

    half_samples: int
    beta: float


#: The true-peak interpolations by name (``method=``), both convolved by FFT:
#: ``"short"`` (``TRUE_PEAK_FILTER_*``), exact to ~0.47 fs; ``"long"``
#: (``TRUE_PEAK_LONG_*``), flat to 0.497 fs.
TRUE_PEAK_METHODS: dict[str, _Interpolation] = {
    "short": _Interpolation(
        TRUE_PEAK_FILTER_HALF_SAMPLES, TRUE_PEAK_FILTER_KAISER_BETA
    ),
    "long": _Interpolation(TRUE_PEAK_LONG_HALF_SAMPLES, TRUE_PEAK_LONG_KAISER_BETA),
}
DEFAULT_TRUE_PEAK_METHOD = "short"


def _interpolation(method: str) -> _Interpolation:
    """The :data:`TRUE_PEAK_METHODS` entry named ``method``."""
    try:
        return TRUE_PEAK_METHODS[method]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown true-peak method {method!r}; one of {sorted(TRUE_PEAK_METHODS)}"
        ) from None


def true_peak_envelope(
    data: np.ndarray,
    fs: int,
    *,
    oversampling: Optional[int] = None,
    method: str = DEFAULT_TRUE_PEAK_METHOD,
) -> np.ndarray:
    """Per-sample true peak ``(n,)``: the largest ``|x|`` over all channels of the
    oversampled signal in ``[n, n+1)``.

    ``oversampling`` defaults to 4x, raised for rates under 48 kHz so the
    oversampled rate is at least ``TRUE_PEAK_MIN_RATE``. ``method`` picks the
    interpolation (:data:`TRUE_PEAK_METHODS`), a windowed sinc convolved by
    FFT. ``"short"`` (default) is 32 samples a side (``TRUE_PEAK_FILTER_*``):
    content up to about 0.47 fs reads as an ideal sinc reconstruction would
    (on the same oversampled grid), content between there and Nyquist reads
    low: ~0.35 dB for noise up to Nyquist, 0.2-1 dB for hard-clipped full-band
    material. ``"long"`` is 2048 samples a side (``TRUE_PEAK_LONG_*``), flat to
    0.497 fs: hard-clipped full-band material reads within ~0.05 dB, at about
    the cost of ``"short"``. The signal is taken as silent outside its span (as
    when played from silence), not looped. Chunked, with the filter's length of
    context shared between chunks, so the joins are exact.
    """
    interpolation = _interpolation(method)
    data = _as_2d(data)
    if oversampling is None:
        factor = max(TRUE_PEAK_OVERSAMPLING, -(-TRUE_PEAK_MIN_RATE // fs))
    elif isinstance(oversampling, numbers.Integral) and oversampling >= 1:
        factor = int(oversampling)
    else:
        raise ValueError(f"oversampling must be an integer >= 1, got {oversampling!r}")
    step = max(1, int(round(FILTER_CHUNK_S * fs)))
    margin = interpolation.half_samples
    envelope = np.empty(len(data))
    for start in range(0, len(data), step):
        stop = min(start + step, len(data))
        lo, hi = max(0, start - margin), min(len(data), stop + margin)
        up = _upsample(
            data[lo:hi],
            factor,
            half_samples=interpolation.half_samples,
            beta=interpolation.beta,
        )
        inner = np.abs(up[(start - lo) * factor : (stop - lo) * factor]).max(axis=1)
        envelope[start:stop] = inner.reshape(-1, factor).max(axis=1)
    return envelope


def true_peak_db(
    data: np.ndarray,
    fs: int,
    *,
    oversampling: Optional[int] = None,
    method: str = DEFAULT_TRUE_PEAK_METHOD,
) -> float:
    """True peak of float samples in dBTP (``-inf`` for silence); ``oversampling``
    and ``method`` as for :func:`true_peak_envelope`."""
    data = _as_2d(data)
    if len(data) == 0:
        return -np.inf
    envelope = true_peak_envelope(data, fs, oversampling=oversampling, method=method)
    peak = float(envelope.max())
    return float(20.0 * np.log10(peak)) if peak > 0 else -np.inf


# --- group leveling -------------------------------------------------------------


def _join(segments: Sequence[np.ndarray], fs: int, min_measure_s: float) -> np.ndarray:
    """Segments back to back, keeping their channels when all have the same
    count (BS.1770 weights channels; a downmix of uncorrelated stereo reads
    ~3 dB low), else downmixed to mono; looped up to ``min_measure_s``."""
    segments = [_as_2d(np.asarray(s, dtype=np.float64)) for s in segments]
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
    the measurement. Silence (at or below the -70 LUFS gate) gets gain 0. A gain
    above ``max_gain_db`` is capped, with a warning, since it would mostly
    amplify noise. Apply it with ``data * 10 ** (gain_db / 20)`` and
    :func:`limit_peaks`.
    """
    if min_measure_s < BLOCK_S:
        raise ValueError(f"min_measure_s must be >= one gate block ({BLOCK_S} s)")
    measured = integrated_lufs(_join(segments, fs, min_measure_s), fs, meter=meter)
    gain = _target_gain(
        measured, target_lufs=target_lufs, max_gain_db=max_gain_db, subject="group"
    )
    return gain, measured


def _target_gain(
    measured: float, *, target_lufs: float, max_gain_db: float, subject: str
) -> float:
    """The gain (dB) from ``measured`` to ``target_lufs``: 0 for silence, capped
    at ``max_gain_db`` with a warning (to the caller's caller) naming ``subject``."""
    if not np.isfinite(target_lufs):
        raise ValueError(f"target_lufs must be a finite LUFS value, got {target_lufs}")
    if np.isnan(max_gain_db):
        raise ValueError("max_gain_db is NaN")
    if not measured > ABSOLUTE_GATE_LUFS:  # silence (or NaN): nothing to level
        return 0.0
    gain = target_lufs - measured
    if gain > max_gain_db:
        warnings.warn(
            f"{subject} measures {measured:.1f} LUFS: reaching {target_lufs:g} would "
            f"take {gain:.1f} dB, mostly noise; capped at +{max_gain_db:g} dB",
            stacklevel=3,
        )
        gain = max_gain_db
    return float(gain)


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
    window: Optional[int] = None,
    true_peak: bool = False,
    fs: Optional[int] = None,
    oversampling: Optional[int] = None,
    method: Optional[str] = None,
) -> np.ndarray:
    """``data`` ``(n, channels)`` with no peak above ``ceiling`` (linear, full scale 1.0).

    The gain each sample needs (``ceiling / peak``, at most 1) is spread over
    ``window`` samples on both sides -- a sliding minimum over ``2 * window +
    1``, then a moving average over ``window`` -- so it eases into and out of
    a peak. Every sample's gain is at most what that sample needs, so the
    ceiling holds exactly; nothing moves in time. ``window`` is in samples;
    by default ``DEFAULT_LIMITER_WINDOW_S`` * ``fs`` when ``fs`` is given, else 1.
    Returned unchanged (the same object, for 2-D input) when nothing exceeds
    the ceiling.
    Works in ``LIMITER_CHUNK_FRAMES`` chunks sharing the context each gain
    depends on, so the result is the whole-signal one at a bounded memory.

    With ``true_peak=True`` (needs ``fs``) the need comes from the oversampled
    peak (see :func:`true_peak_envelope`) of each sample and the gap before it,
    so inter-sample peaks are held at ``ceiling`` too, up to the small error of
    interpolating the gain between samples. That holds when ``window`` is a few
    milliseconds or more (the default is 50 ms); a window of a sample or two
    can overshoot by ~1 dB because the gain then changes faster than the
    interpolation assumes.

    ``oversampling`` (with ``true_peak=True`` only) is the true-peak meter's
    factor (default: the meter's, 4x at 48 kHz, raised below 48 kHz; an
    explicit factor is used as given). The peak *between* grid points is not
    seen, so the true
    ceiling holds to the grid: limited to -1 dBTP, bright material (noise up
    to 0.45 fs) measures about -0.8 dBTP on a fine (32x) grid at 4x, -0.97
    at 8x, -0.99 at 16x, at roughly 1x / 2x / 4x the cost. Content between
    0.47 fs and Nyquist reads low whatever the factor with the default
    ``method="short"`` (see :func:`true_peak_envelope`): limited to -1 dBTP,
    hard-clipped full-band material measures -0.2 to -0.7 dBTP, so set the
    ceiling ~1 dB under the target. ``method="long"`` (with ``true_peak=True``
    only; the meter's interpolation) reads that content too, and holds the
    -1 dBTP ceiling on clipped material to ~0.06 dB whatever the factor
    (clipped noise peaks midway between samples, on every even grid); on
    band-limited bright material the grid still matters as above.
    """
    margin, block_gain = _limiter(
        ceiling=ceiling,
        window=window,
        true_peak=true_peak,
        fs=fs,
        oversampling=oversampling,
        method=method,
    )
    data = _as_2d(data)
    n, out = len(data), None
    for start in range(0, n, LIMITER_CHUNK_FRAMES):
        stop = min(n, start + LIMITER_CHUNK_FRAMES)
        lo, hi = max(0, start - margin), min(n, stop + margin)
        gain = block_gain(data[lo:hi], start - lo, stop - lo)
        if out is None:
            if gain.min() >= 1.0:
                continue  # untouched so far
            out = data.astype(np.result_type(data.dtype, np.float64))
        out[start:stop] = data[start:stop] * gain[:, None]
    return data if out is None else out


def stream_limit_peaks(
    chunks: Iterable[np.ndarray],
    *,
    ceiling: float,
    fs: Optional[int] = None,
    window: Optional[int] = None,
    true_peak: bool = False,
    oversampling: Optional[int] = None,
    method: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """:func:`limit_peaks` over a signal that arrives in chunks.

    ``chunks`` are float arrays ``(k, channels)`` (or 1-D, mono), any sizes,
    all with the same channel count; the arguments are :func:`limit_peaks`'.
    Yields the limited signal (float64) in blocks of ``LIMITER_CHUNK_FRAMES``
    frames, the last one shorter, equal to :func:`limit_peaks` on the joined
    signal. It holds one block plus the context each gain depends on (twice the
    window, and the true-peak filter's length) and the chunk being read, so a
    track of any length is limited in bounded memory. The settings are
    checked here, before the first chunk is read.
    """
    margin, block_gain = _limiter(
        ceiling=ceiling,
        window=window,
        true_peak=true_peak,
        fs=fs,
        oversampling=oversampling,
        method=method,
    )
    return _limited_blocks(chunks, margin, block_gain)


def _limited_blocks(chunks, margin: int, block_gain) -> Iterator[np.ndarray]:
    """:func:`stream_limit_peaks`' generator: the blocks of :func:`limit_peaks`'
    grid, each computed once the context ``margin`` past its end has arrived."""
    held, held_start, done = None, 0, 0  # ``held`` is frames [held_start, ...)

    def block():
        nonlocal held, held_start, done
        end = held_start + len(held)
        start, stop = done, min(end, done + LIMITER_CHUNK_FRAMES)
        lo, hi = max(0, start - margin), min(end, stop + margin)
        gain = block_gain(
            held[lo - held_start : hi - held_start], start - lo, stop - lo
        )
        out = held[start - held_start : stop - held_start] * gain[:, None]  # float64
        done = stop
        keep = max(0, done - margin)
        held, held_start = held[keep - held_start :], keep
        return out

    for chunk in chunks:
        chunk = _as_2d(np.asarray(chunk))
        if not len(chunk):
            continue
        if held is not None and chunk.shape[1] != held.shape[1]:
            raise ValueError(
                f"chunks must all have {held.shape[1]} channels, got {chunk.shape[1]}"
            )
        held = chunk if held is None else np.concatenate([held, chunk])
        while held_start + len(held) >= done + LIMITER_CHUNK_FRAMES + margin:
            yield block()
    while held is not None and done < held_start + len(held):
        yield block()


def _limiter(*, ceiling, window, true_peak, fs, oversampling, method):
    """``(margin, block_gain)`` for these (validated) settings.

    ``block_gain(context, start, stop)`` is the gain of ``context[start:stop]``:
    the whole-signal one when ``context`` reaches ``margin`` samples past
    either side of the block, or the signal's edge.
    """
    if not 0 < ceiling <= 1:
        raise ValueError(f"ceiling is linear, in (0, 1]; got {ceiling} (not dB)")
    for name, value in (("oversampling", oversampling), ("method", method)):
        if value is not None and not true_peak:
            raise ValueError(f"{name} applies to true_peak=True only")
    if true_peak and fs is None:
        raise ValueError("true_peak=True needs fs")
    if oversampling is not None and not (
        isinstance(oversampling, numbers.Integral) and oversampling >= 1
    ):
        raise ValueError(f"oversampling must be an integer >= 1, got {oversampling!r}")
    if method is None:
        method = DEFAULT_TRUE_PEAK_METHOD
    interpolation = _interpolation(method)
    if window is None:
        window = round(DEFAULT_LIMITER_WINDOW_S * fs) if fs else 1
    w = max(1, int(window))
    # a gain sees the need up to ~1.5w samples away on each side (sliding
    # minimum, then moving average); a sample's true-peak need reads the
    # filter's half-length of samples on each side (its envelope reads
    # [i - half + 1, i + half], the gap before it one more on the left), and
    # the + 1 is one sample of slack
    margin = 2 * w + 1 + (interpolation.half_samples + 1 if true_peak else 0)

    def block_gain(context: np.ndarray, start: int, stop: int) -> np.ndarray:
        need = _peak_need(
            context,
            ceiling=ceiling,
            true_peak=true_peak,
            fs=fs,
            oversampling=oversampling,
            method=method,
        )
        return _smoothed_gain(need, w)[start:stop]

    return margin, block_gain


def _peak_need(
    data: np.ndarray, *, ceiling: float, true_peak: bool, fs, oversampling, method
) -> np.ndarray:
    """The gain each sample needs, ``ceiling / peak`` at most 1 ``(n,)``: by its
    sample peak, or by the true peak of it and the gap before it."""
    if true_peak:
        peak = true_peak_envelope(data, fs, oversampling=oversampling, method=method)
        peak = np.maximum(peak, np.concatenate([[0.0], peak[:-1]]))  # the gap before
    else:
        peak = np.abs(data).max(axis=1)
    return np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))


def _smoothed_gain(need: np.ndarray, w: int) -> np.ndarray:
    """``need`` eased in and out over ``w`` samples (sliding minimum over
    ``2w + 1``, then a moving average over ``w``), never above ``need``."""
    if need.min() >= 1.0:
        return need
    lows = _sliding_min(np.pad(need, (w, w), constant_values=1.0), 2 * w + 1)
    padded = np.pad(lows, (w // 2, w - 1 - w // 2), mode="edge")
    gain = np.convolve(padded, np.ones(w) / w, mode="valid")
    return np.minimum(gain, need)  # float rounding never lets a peak through


# --- a file to a loudness target ---------------------------------------------------


class LeveledFile(NamedTuple):
    """What :func:`level_file` did: where it wrote, the gain, and the loudness
    before and after (``output_lufs`` falls short of the target when the
    limiter had to take a lot off, or the gain was capped)."""

    output: (
        Any  #: what the ``output`` protocol returned (the ``Path``, or a sink's result)
    )
    gain_db: float
    measured_lufs: float  #: the source's integrated loudness, at the output rate
    output_lufs: float  #: the written file's, measured back from it


def level_file(
    src,
    output: Output = None,
    *,
    target_lufs: float,
    ceiling_db: float = DEFAULT_CEILING_DB,
    true_peak: bool = True,
    sample_rate: Optional[int] = None,
    max_gain_db: float = MAX_GAIN_DB,
    method: Optional[str] = None,
    oversampling: Optional[int] = None,
    window_s: Optional[float] = None,
    encode_args: Sequence[str] = (),
    chunk_s: float = DECODE_CHUNK_S,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LeveledFile:
    """Write ``src``'s audio at ``target_lufs`` with one measured gain, peak limited.

    Two passes over the file, each decoded in chunks by
    :func:`mixing.decode.decode_float_chunks` (``chunk_s``, ``ffmpeg``;
    ``timeout`` bounds each ffmpeg run, see :mod:`mixing.decode`: four of
    them, five in place, so the whole call is bounded by about five times
    it), so
    memory is bounded by the chunk and the limiter's block, not the track:

    1. :func:`stream_loudness` measures it; the gain is the one that brings it
       to ``target_lufs`` (as :func:`level_group`'s: 0 for silence, capped at
       ``max_gain_db`` with a warning), so the audio keeps its own dynamics,
       unlike ffmpeg ``loudnorm``, which rides the gain through it;
    2. the gain is applied in float (nothing clips before the limiter),
       :func:`stream_limit_peaks` holds ``ceiling_db`` (dBTP with
       ``true_peak``, else dBFS; ``method``, ``oversampling`` and the
       lookahead ``window_s``, in seconds, are its), and
       :func:`mixing.decode.encode_float_chunks` writes it (``encode_args``
       are extra ffmpeg output options, such as ``("-c:a", "pcm_s24le")``).

    The written file is then metered back (``output_lufs``): it reads below
    the target when the limiter took a lot off. ``sample_rate`` resamples as
    the file is decoded, so the measurement and the limiter work at the rate
    written and the ceiling holds in the output (an encoder resampling after
    the limiter could push a peak over it). A lossy format (mp3, aac) can
    still read a few tenths of a dB over the ceiling.

    The output is **audio only** (the first audio stream; no picture, cover
    art or tags). ``output`` follows mixing's ``output`` protocol (default:
    beside ``src`` with ``LEVELED_SUFFIX`` on its stem). It may be ``src``
    itself, replaced once the new file is complete, unless ``src`` has other
    streams (a video's picture, say), which that would destroy: a
    ``ValueError`` then.

    Returns a :class:`LeveledFile` ``(output, gain_db, measured_lufs, output_lufs)``.
    """
    src = Path(src)
    if not -np.inf < ceiling_db <= 0:
        raise ValueError(
            f"ceiling_db is in dB, at or below full scale; got {ceiling_db}"
        )
    if not np.isfinite(target_lufs):
        raise ValueError(f"target_lufs must be a finite LUFS value, got {target_lufs}")
    if window_s is not None and not 0 < window_s < np.inf:
        raise ValueError(f"window_s is in seconds, > 0; got {window_s}")
    ceiling = 10.0 ** (ceiling_db / 20.0)
    limits = dict(true_peak=true_peak, oversampling=oversampling, method=method)
    # the settings are checked before the first pass (any rate does for that)
    _limiter(ceiling=ceiling, window=None, fs=sample_rate or 1, **limits)
    decode = partial(
        decode_float_chunks,
        src,
        chunk_s=chunk_s,
        ffmpeg=ffmpeg,
        sample_rate=sample_rate,
        timeout=timeout,
    )
    rate, chunks = decode()
    with closing(chunks):
        measured = stream_loudness(chunks, rate).integrated_lufs
    gain_db = _target_gain(
        measured, target_lufs=target_lufs, max_gain_db=max_gain_db, subject=src.name
    )
    scale = 10.0 ** (gain_db / 20.0)
    output_lufs = None

    def write(path: Path) -> None:
        nonlocal output_lufs
        # samefile, not resolve(): a case-insensitive filesystem (macOS) spells
        # one file two ways
        in_place = path.exists() and os.path.samefile(path, src)
        kinds = stream_kinds(src, ffmpeg=ffmpeg, timeout=timeout) if in_place else []
        if others := sorted({kind for kind in kinds if kind != "audio"}):
            raise ValueError(
                f"leveling {src.name} in place would drop its {', '.join(others)} "
                "stream(s): level_file writes audio only; give another output"
            )
        rate, chunks = decode()
        window = None if window_s is None else max(1, round(window_s * rate))
        with closing(chunks):
            leveled = stream_limit_peaks(
                (chunk * scale for chunk in chunks),
                ceiling=ceiling,
                fs=rate,
                window=window,
                **limits,
            )
            encode_float_chunks(
                leveled, rate, path, args=encode_args, ffmpeg=ffmpeg, timeout=timeout
            )
        output_lufs = file_loudness(
            path, chunk_s=chunk_s, ffmpeg=ffmpeg, timeout=timeout
        ).integrated_lufs

    default_path = src.with_name(f"{src.stem}{LEVELED_SUFFIX}{src.suffix}")
    written = write_egress(output, default_path=default_path, write=write)
    return LeveledFile(written, gain_db, measured, output_lufs)
