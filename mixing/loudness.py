"""Array-only loudness: meters, group leveling and a peak limiter.

Everything here takes plain float arrays shaped ``(n, channels)`` in [-1, 1]
plus a sample rate, and needs only numpy and scipy (extra ``loudness``,
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
  in one pass over chunks (or a file decoded in chunks), in memory bounded by
  the chunk, not the track.

>>> import numpy as np
>>> integrated_lufs(np.zeros((100, 1)), 8000)
-inf
>>> limit_peaks(np.array([[2.0], [0.5]]), ceiling=1.0)[:, 0].tolist()
[1.0, 0.25]
"""

from __future__ import annotations

import numbers
import re
import struct
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Sequence, Union

from mixing.util import LOUDNESS_EXTRA, ffmpeg_exe, install_hint

try:  # numpy only: scipy is imported where used, so the ffmpeg meter works without it
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"mixing.loudness needs numpy and scipy: {install_hint(LOUDNESS_EXTRA)}",
        name=e.name,
    ) from e

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
    from scipy.signal import lfilter

    sections, states = _k_weighting_coefficients(fs), None
    for x in chunks:
        x = _as_2d(np.asarray(x, dtype=np.float64))
        if states is None:  # biquads: order 2
            states = [np.zeros((2, x.shape[1])) for _ in sections]
        for k, (b, a) in enumerate(sections):
            x, states[k] = lfilter(b, a, x, axis=0, zi=states[k])
        yield (x**2).sum(axis=1)


@dataclass
class _Blocks:
    """Mean square of every ``block``-long window at ``hop`` spacing, over
    energy fed in pieces; holds at most a block plus one piece. A signal
    shorter than one block is one block over all of it."""

    block: int
    hop: int

    def __post_init__(self):
        self._buffer, self._offset, self._next, self._out = np.zeros(0), 0, 0, []
        self.length = 0

    def feed(self, energy: np.ndarray) -> None:
        self.length += len(energy)
        self._buffer = np.concatenate([self._buffer, energy])
        end = self._offset + len(self._buffer)
        starts = np.arange(self._next, end - self.block + 1, self.hop)
        if len(starts):
            cumulative = np.concatenate([[0.0], np.cumsum(self._buffer)])
            at = starts - self._offset
            self._out.append(
                (cumulative[at + self.block] - cumulative[at]) / self.block
            )
            self._next = int(starts[-1]) + self.hop
            drop = min(self._next - self._offset, len(self._buffer))
            self._buffer, self._offset = self._buffer[drop:], self._offset + drop

    def result(self) -> np.ndarray:
        if self._out:
            return np.concatenate(self._out)
        return self._buffer[None].mean(axis=1) if self.length else np.zeros(0)


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


#: ffmpeg decodes a file for :func:`file_loudness` as 32-bit float WAV...
_WAV_FLOAT = "pcm_f32le"
#: ...read this many seconds at a time: a 30 min stereo track meters at ~130 MB
#: peak RSS (~106 MB of it numpy and scipy), the same as a 5 min one.
FILE_CHUNK_S = 5.0


def _wav_format(stream) -> tuple[int, int]:
    """``(channels, rate)`` from a streamed WAV header, read up to its ``data``
    chunk (ffmpeg writes unknown sizes when piping, so they are not used)."""
    header = stream.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError("ffmpeg did not produce a WAV stream")
    channels = rate = None
    while True:
        chunk = stream.read(8)
        if len(chunk) < 8:
            raise RuntimeError("WAV stream ended before its data chunk")
        name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
        if name == b"data":
            if channels is None:
                raise RuntimeError("WAV stream has no fmt chunk")
            return channels, rate
        body = stream.read(size + size % 2)
        if name == b"fmt ":
            channels, rate = struct.unpack("<HI", body[2:8])


def file_loudness(
    path, *, chunk_s: float = FILE_CHUNK_S, ffmpeg: Optional[str] = None
) -> LoudnessReading:
    """Integrated loudness and LRA of a media file's audio, in bounded memory.

    ffmpeg (``ffmpeg``, default :func:`mixing.util.ffmpeg_exe`) decodes the
    first audio stream at its own rate and channel count (no resampling, no
    downmix) as float WAV, read ``chunk_s`` seconds at a time into
    :func:`stream_loudness`; memory stays a chunk's worth whatever the
    length. Raises ``RuntimeError`` when ffmpeg cannot decode it.
    """
    cmd = [
        ffmpeg or ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0",
        "-c:a", _WAV_FLOAT, "-f", "wav", "-",
    ]  # fmt: skip
    # stderr to a file: a pipe nobody reads fills up and blocks ffmpeg
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors) as proc:
            try:
                channels, rate = _wav_format(proc.stdout)
            except RuntimeError:
                channels = None
            if channels is not None:
                frame = 4 * channels
                step = max(1, int(round(chunk_s * rate))) * frame

                def chunks() -> Iterator[np.ndarray]:
                    while block := proc.stdout.read(step):
                        usable = len(block) - len(block) % frame
                        yield np.frombuffer(block[:usable], "<f4").reshape(-1, channels)

                reading = stream_loudness(chunks(), rate)
        errors.seek(0)
        message = errors.read().decode(errors="replace")
    if proc.returncode != 0 or channels is None:
        raise RuntimeError(f"ffmpeg could not decode the audio of {path!s}: {message}")
    return reading


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
def _interpolation_filter(factor: int) -> np.ndarray:
    """The windowed-sinc FIR that upsamples by ``factor``, scaled for
    ``resample_poly`` (which multiplies by ``factor``). Its every ``factor``-th
    tap is zero but the centre one, so the original samples pass unchanged and
    a true peak never reads below the sample peak. Read-only: it is cached."""
    from scipy.signal.windows import kaiser

    half = TRUE_PEAK_FILTER_HALF_SAMPLES * factor
    n = np.arange(-half, half + 1)
    taps = np.sinc(n / factor) * kaiser(len(n), TRUE_PEAK_FILTER_KAISER_BETA)
    taps[n % factor == 0] = 0.0  # exactly, not sinc's ~1e-17
    taps[half] = 1.0
    taps /= factor
    taps.flags.writeable = False
    return taps


def true_peak_envelope(
    data: np.ndarray, fs: int, *, oversampling: Optional[int] = None
) -> np.ndarray:
    """Per-sample true peak ``(n,)``: the largest ``|x|`` over all channels of the
    oversampled signal in ``[n, n+1)``.

    ``oversampling`` defaults to 4x, raised for rates under 48 kHz so the
    oversampled rate is at least ``TRUE_PEAK_MIN_RATE``. The interpolation is a
    long windowed sinc (``TRUE_PEAK_FILTER_*``): content up to about 0.47 fs
    reads as an ideal sinc reconstruction would (on the same oversampled grid),
    content between there and Nyquist reads low: ~0.35 dB for noise up to
    Nyquist, ~1 dB for hard-clipped full-band material. The signal is
    taken as silent outside its span (as when played from silence), not
    looped. Chunked, with the filter's length of context shared between chunks,
    so the joins are exact.
    """
    from scipy.signal import resample_poly

    data = _as_2d(data)
    if oversampling is None:
        factor = max(TRUE_PEAK_OVERSAMPLING, -(-TRUE_PEAK_MIN_RATE // fs))
    elif isinstance(oversampling, numbers.Integral) and oversampling >= 1:
        factor = int(oversampling)
    else:
        raise ValueError(f"oversampling must be an integer >= 1, got {oversampling!r}")
    taps = _interpolation_filter(factor)
    step = max(1, int(round(FILTER_CHUNK_S * fs)))
    margin = TRUE_PEAK_FILTER_HALF_SAMPLES
    envelope = np.empty(len(data))
    for start in range(0, len(data), step):
        stop = min(start + step, len(data))
        lo, hi = max(0, start - margin), min(len(data), stop + margin)
        up = resample_poly(data[lo:hi], factor, 1, axis=0, window=taps)
        inner = np.abs(up[(start - lo) * factor : (stop - lo) * factor]).max(axis=1)
        envelope[start:stop] = inner.reshape(-1, factor).max(axis=1)
    return envelope


def true_peak_db(
    data: np.ndarray, fs: int, *, oversampling: Optional[int] = None
) -> float:
    """True peak of float samples in dBTP (``-inf`` for silence)."""
    data = _as_2d(data)
    if len(data) == 0:
        return -np.inf
    peak = float(true_peak_envelope(data, fs, oversampling=oversampling).max())
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
    if not measured > ABSOLUTE_GATE_LUFS:  # silence (or NaN): nothing to level
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
    window: Optional[int] = None,
    true_peak: bool = False,
    fs: Optional[int] = None,
    oversampling: Optional[int] = None,
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
    to 0.45 fs) measures about -0.8 dBTP on a fine (32x) grid at 4x, -0.96
    at 8x, -0.99 at 16x, at roughly 1x / 2x / 4x the cost. Content between
    0.47 fs and Nyquist reads low whatever the factor (see
    :func:`true_peak_envelope`): ~0.35 dB for noise up to Nyquist, ~1 dB for
    hard-clipped full-band material. For a hard ceiling on such material use
    ``oversampling=8`` and a ceiling ~1 dB under the target.
    """
    if not 0 < ceiling <= 1:
        raise ValueError(f"ceiling is linear, in (0, 1]; got {ceiling} (not dB)")
    if oversampling is not None and not true_peak:
        raise ValueError("oversampling applies to true_peak=True only")
    if true_peak and fs is None:
        raise ValueError("true_peak=True needs fs")
    data = _as_2d(data)
    if window is None:
        window = round(DEFAULT_LIMITER_WINDOW_S * fs) if fs else 1
    w = max(1, int(window))
    # a gain sees the need up to ~1.5w samples away on each side (sliding
    # minimum, then moving average); a true peak, the interpolation filter's
    # length of samples and the one before
    margin = 2 * w + 1 + (TRUE_PEAK_FILTER_HALF_SAMPLES + 1 if true_peak else 0)
    n, out = len(data), None
    for start in range(0, n, LIMITER_CHUNK_FRAMES):
        stop = min(n, start + LIMITER_CHUNK_FRAMES)
        lo, hi = max(0, start - margin), min(n, stop + margin)
        need = _peak_need(
            data[lo:hi],
            ceiling=ceiling,
            true_peak=true_peak,
            fs=fs,
            oversampling=oversampling,
        )
        gain = _smoothed_gain(need, w)[start - lo : stop - lo]
        if out is None:
            if gain.min() >= 1.0:
                continue  # untouched so far
            out = data.astype(np.result_type(data.dtype, np.float64))
        out[start:stop] = data[start:stop] * gain[:, None]
    return data if out is None else out


def _peak_need(
    data: np.ndarray, *, ceiling: float, true_peak: bool, fs, oversampling
) -> np.ndarray:
    """The gain each sample needs, ``ceiling / peak`` at most 1 ``(n,)``: by its
    sample peak, or by the true peak of it and the gap before it."""
    if true_peak:
        peak = true_peak_envelope(data, fs, oversampling=oversampling)
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
