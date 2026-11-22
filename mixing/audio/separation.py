"""Separate speech from music, and remove a host's talk while keeping the music.

Two layers, each usable on its own:

- :func:`separate_speech` ``(audio) -> (speech, music)``: the **separation
  seam**. ``separator=`` is a key of :data:`SEPARATORS`, a
  :class:`SeparationStrategy`, or any callable ``(samples (n, ch), sr) ->
  (speech, music)``, so a model mixing does not ship (Bandit-v2 from ZFTurbo's
  MSST repo, a host-fine-tuned network) plugs in without mixing depending on it.
- :func:`remove_talk` ``(audio, spans) -> audio``: the **splice combinator**. It
  runs the separator only on the given talk spans (widened by ``margin``, plus
  model context), fades the music estimate in and out linearly over the margin,
  and leaves every sample outside the widened spans identical to the input. The
  spans can come from anywhere: :class:`~mixing.audio.talk_edges.TalkEdges` (from
  ``find_talk_edges``), ``Segment`` s, or ``(start, end)`` pairs in seconds.

A :class:`SeparationStrategy` wraps a model with what the framework needs to run
it: its native sample rate (audio is resampled in and out), whether it is
mono-only (then it runs once per channel: radio music is wide and the talk sits
near the centre, so per-channel runs matter), how much context it needs around a
span, the mixing extra that installs it, and its licence (``commercial_ok`` lets
a caller filter).

**A lower-rate model passes the high band through as "music".** When a
strategy's rate is below the input's, the band above the model's Nyquist
frequency (8 kHz for a 16 kHz enhancer) never reaches the model, so it is added
to the music untouched, *including any talk there* (sibilants, breath). With the
registered 44.1/48 kHz strategies this does not arise for CD-rate input; it
matters when plugging in a 16 kHz model. What the resampling round trip
weakens where the model did hear it (the filter's slope near the lower of the
two Nyquist frequencies) is split between the stems in proportion to them, and
the images resampling makes of the stems above the model's Nyquist are dropped:
so ``speech + music`` is the input whenever the model's own stems sum to what
it was given (a model that drops a stem still drops it), and talk the model
found stays out of the music, to within about one STFT bin of that line.

Registered strategies (licences are the checkpoints'):

- ``"mossformer2_se_48k"`` (default, Apache-2.0): the MossFormer2-SE-48K speech
  enhancer of ClearVoice, *inverted* by :func:`invert_enhancer`: music is the
  mixture minus the speech estimate, split with a Wiener mask. Keeps singing (it
  is not speech to an enhancer). Install: ``pip install 'mixing[separation]'``,
  then ``pip install --no-deps 'clearvoice==0.1.2'`` (clearvoice's own pins
  would downgrade numpy below 2 and add a second ``cv2``; the adapter uses
  0.1.2's internals, hence the exact pin).
- ``"htdemucs"`` (MIT, ``pip install 'mixing[demucs]'``): the ``vocals`` stem of
  Hybrid Transformer Demucs as the speech. **Instrumental beds only**: it removes
  the singing along with the talk.
- ``"voxy_htdemucs_ft"`` (MIT, ``pip install 'mixing[voxy]'``): the same job by
  voxy's Demucs (``htdemucs_ft``), run **out of process** in voxy's own
  environment, so this process imports no torch (:func:`voxy_separator`).
  Instrumental beds only, as ``"htdemucs"``.

Heavy dependencies (``torch``, ``clearvoice``, ``demucs``) are imported inside the
strategies, so ``import mixing.audio`` stays light, and a missing one raises an
``ImportError`` with its install line. Models load once per process.

>>> import numpy as np
>>> sr = 8000
>>> x = np.full((sr, 1), 0.5)                       # 1 s of a constant "mixture"
>>> halve = lambda a, rate: (a / 2, a / 2)          # a toy separator
>>> out = remove_talk(x, [(0.4, 0.6)], separator=halve, sr=sr, margin=0.1)
>>> float(out[0, 0]), float(out[4000, 0]), float(out[-1, 0])
(0.5, 0.25, 0.5)
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from ..egress import Output, deliver
from ..util import (
    DEMUCS_EXTRA,
    SEPARATION_EXTRA,
    VOXY_EXTRA,
    install_hint,
    require_extra,
)
from ..util import require_package
from ._pcm import resample
from .samples import AudioLike, float_input
from .spans import Span, as_spans

#: The separator seam: ``(float samples (n, channels), sr) -> (speech, music)``,
#: both shaped like the input and at the same rate.
SpeechSeparator = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
#: A speech enhancer: ``(float samples (n, channels), sr) -> speech estimate``.
SpeechEnhancer = Callable[[np.ndarray, int], np.ndarray]

#: Each talk span is widened by this much (seconds) on both sides, and the music
#: estimate fades in (out) linearly over that widening, so the span itself is all
#: music estimate. It absorbs the timing error of the span's producer
#: (``find_talk_edges`` without a transcript is window-resolution) and the soft
#: onset and decay of a voice; 0.3 s is the splice #63's experiment scored.
#: ``margin=0`` is a hard cut at the span's edges.
DEFAULT_MARGIN_S = 0.3
#: Context (seconds) given on each side of a span to a separator that does not say
#: how much it needs (plain callables): enough for a model's receptive field to
#: settle before the span starts.
DEFAULT_CONTEXT_S = 1.0
#: A separator may return up to this much (seconds) more or less than it was
#: given (STFT framing); the result is padded or cut to the exact length.
LENGTH_TOLERANCE_S = 0.05
#: Wiener-mask STFT window (seconds), rounded to a power of two in samples
#: (2048 at 48 kHz): fine enough in frequency to tell harmonics of a voice from
#: those of the music under it.
WIENER_WINDOW_S = 0.04
#: Floor on the Wiener mask's denominator (power), so silent bins do not divide by
#: zero (they get mask 0, which is harmless: both sources are silent there).
WIENER_EPS = 1e-20
#: Frames per block when :func:`_split_round_trip` walks the STFT a block at a
#: time: its peak memory is a few blocks' spectra instead of a few whole
#: signals' (reeleehq/mixing#203). Any value gives the same output.
STFT_BLOCK_FRAMES = 256
#: Suffix of a written result's name (``<stem>_talk_removed.wav``).
OUTPUT_SUFFIX = "_talk_removed.wav"

#: Hugging Face repo of the MossFormer2-SE-48K checkpoint (Apache-2.0, ~220 MB).
MOSSFORMER2_SE_48K_REPO = "alibabasglab/MossFormer2_SE_48K"
#: The model name ClearVoice (clearvoice) knows it by.
MOSSFORMER2_SE_48K_NAME = "MossFormer2_SE_48K"
#: Native sample rate of MossFormer2-SE-48K.
MOSSFORMER2_SE_48K_RATE = 48000
#: Context (seconds) around a span for MossFormer2-SE: its STFT frames are 40 ms
#: and its sequence model sees the whole input, so a second each side suffices.
MOSSFORMER2_SE_48K_CONTEXT_S = 1.0
#: The clearvoice release whose internals the adapter calls (``network_wrapper.
#: load_args_se``, ``CLS_MossFormer2_SE_48K``, ``utils.decode.decode_one_audio``).
CLEARVOICE_VERSION = "0.1.2"
#: How to install clearvoice without its pins (numpy<2, opencv-python, ...).
CLEARVOICE_INSTALL_HINT = (
    f"{install_hint(SEPARATION_EXTRA)}, then "
    f"pip install --no-deps 'clearvoice=={CLEARVOICE_VERSION}'"
)
#: clearvoice 0.1.2's windowed decode (inputs above ``one_time_decode_length``)
#: hops by this fraction of its window; hard-coded there, mirrored here.
CLEARVOICE_STRIDE_FRACTION = 0.75
#: The pretrained Demucs model used by the ``"htdemucs"`` strategy.
HTDEMUCS_MODEL = "htdemucs"
#: Native sample rate of htdemucs.
HTDEMUCS_RATE = 44100
#: The htdemucs stem taken as the speech.
HTDEMUCS_SPEECH_STEM = "vocals"
#: Context (seconds) around a span for htdemucs: it separates ~7.8 s segments
#: with overlap, so give it a few seconds of the music on either side.
HTDEMUCS_CONTEXT_S = 3.0
#: Added to the standard deviation when htdemucs' input is standardised (as
#: ``demucs.api.Separator`` does), so digital silence does not divide by zero.
HTDEMUCS_STD_EPS = 1e-8


@dataclass(frozen=True)
class SeparationStrategy:
    """A speech/music separator plus what the framework needs to run it.

    Calling the strategy, ``strategy(samples, sr)``, runs it the way
    :func:`separate_speech` does: it checks the extra, resamples to
    ``sample_rate`` and back, runs per channel if ``mono``, and fits the outputs
    to the input's shape. ``separate`` itself only ever sees audio at its native
    rate (``(n, 1)`` if ``mono``).

    Attributes:
        name: Registry key / display name (errors name it).
        separate: ``(samples (n, ch), sr) -> (speech, music)`` at ``sample_rate``.
        license: Licence of the weights (and code) the strategy runs.
        commercial_ok: Whether that licence allows commercial use (``None``:
            unknown, as for a caller's bare callable).
        extra: The mixing extra that installs the strategy's dependencies, or
            ``None``; checked before the strategy runs.
        mono: True if the model takes one channel: it then runs per channel.
        sample_rate: The model's native rate, or ``None`` for any rate. Below
            the input's rate, the band above its Nyquist frequency goes to the
            music untouched (talk included).
        context_s: Seconds of audio the model wants around a span
            (:func:`remove_talk` cuts this much context on each side).
    """

    name: str
    separate: SpeechSeparator = field(repr=False)
    license: Optional[str] = None
    commercial_ok: Optional[bool] = None
    extra: Optional[str] = None
    mono: bool = False
    sample_rate: Optional[int] = None
    context_s: float = DEFAULT_CONTEXT_S

    def __call__(self, samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        return _run_strategy(self, samples, sr)


# ---------------------------------------------------------------------------
# The framework: resampling, per-channel runs, length fitting
# ---------------------------------------------------------------------------


def _fit(out: Any, n: int, channels: int, *, sr: int, what: str) -> np.ndarray:
    """``out`` as float ``(n, channels)``: 1-D is one channel, the length is padded
    or cut to ``n`` within :data:`LENGTH_TOLERANCE_S`."""
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[1] != channels:
        raise ValueError(f"{what} must have shape (n, {channels}), got {out.shape}")
    if abs(len(out) - n) > max(1, int(round(LENGTH_TOLERANCE_S * sr))):
        raise ValueError(
            f"{what} has {len(out)} frames for {n} "
            f"(tolerance LENGTH_TOLERANCE_S={LENGTH_TOLERANCE_S}s)"
        )
    out = out[:n]
    return np.pad(out, ((0, n - len(out)), (0, 0))) if len(out) < n else out


def _run_native(strategy: SeparationStrategy, x: np.ndarray, sr: int):
    """``strategy.separate`` on ``x`` (already at its rate), per channel if mono."""
    n, ch = x.shape

    def run(block):
        speech, music = strategy.separate(block, sr)
        width = block.shape[1]
        return (
            _fit(speech, n, width, sr=sr, what="the speech"),
            _fit(music, n, width, sr=sr, what="the music"),
        )

    try:
        if not strategy.mono or ch == 1:
            return run(x)
        pairs = [run(x[:, [c]]) for c in range(ch)]
    except ValueError as e:
        raise ValueError(f"separator {strategy.name!r}: {e}") from e
    return tuple(np.hstack([p[i] for p in pairs]) for i in (0, 1))


def _run_strategy(
    strategy: SeparationStrategy, x: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``strategy`` on float ``x`` ``(n, ch)`` at ``sr``: ``(speech, music)``
    shaped like ``x``, at ``sr``."""
    if strategy.extra is not None:
        require_extra(f"the {strategy.name!r} separator", strategy.extra)
    x = np.asarray(x, dtype=np.float64)
    n, ch = x.shape
    if n == 0:
        return x.copy(), x.copy()
    native = strategy.sample_rate or sr
    if native == sr:
        return _run_native(strategy, x, sr)
    x_native = resample(x, sr, native)
    speech, music = _run_native(strategy, x_native, native)

    def back(y):
        what = f"separator {strategy.name!r}'s resampled output"
        return _fit(resample(y, native, sr), n, ch, sr=sr, what=what)

    return _split_round_trip(
        x,
        back(speech),
        back(music),
        x - back(x_native),
        sr,
        heard_hz=None if native >= sr else native / 2,  # None: it heard it all
    )


def _resolve(separator: Union[str, SeparationStrategy, SpeechSeparator]):
    """``separator`` as a :class:`SeparationStrategy`."""
    if isinstance(separator, SeparationStrategy):
        return separator
    if isinstance(separator, str):
        try:
            return SEPARATORS[separator]
        except KeyError:
            raise ValueError(
                f"unknown separator {separator!r}; one of {sorted(SEPARATORS)}, "
                "or pass a callable (samples, sr) -> (speech, music)"
            ) from None
    if callable(separator):
        return SeparationStrategy(
            name=getattr(separator, "__name__", repr(separator)), separate=separator
        )
    raise TypeError(
        f"separator must be a registry key, a SeparationStrategy or a callable, "
        f"got {type(separator).__name__}"
    )


# ---------------------------------------------------------------------------
# Inverting a speech enhancer (mixture minus speech, Wiener mask)
# ---------------------------------------------------------------------------


def _stft_size(sr: int, window_s: float) -> int:
    """The power of two nearest ``window_s * sr`` samples (at least 16)."""
    return max(16, 1 << int(round(np.log2(max(1.0, window_s * sr)))))


def wiener_split(
    mixture: np.ndarray,
    speech: np.ndarray,
    sr: int,
    *,
    window_s: float = WIENER_WINDOW_S,
) -> tuple[np.ndarray, np.ndarray]:
    """``(speech, music)`` from a mixture and a speech estimate, by Wiener masking.

    The music estimate is the residual ``mixture - speech``; in every STFT bin the
    mixture is split by the power ratio of the two estimates (music gets
    ``|M|^2 / (|M|^2 + |S|^2)``, in [0, 1]). The two outputs sum to the mixture,
    and neither is louder than it in any bin, however wrong the estimate (plain
    subtraction is not: an enhancer returning ``-x`` would double the music).

    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(4000, 1))
    >>> s, m = wiener_split(x, 0.5 * x, 8000)
    >>> bool(np.allclose(s + m, x))
    True
    """
    mixture = np.asarray(mixture, dtype=np.float64)
    speech = np.asarray(speech, dtype=np.float64)
    n = len(mixture)
    spec, inverse, _ = _stft_pair(sr, n, window_s=window_s)
    X, S = spec(mixture), spec(speech)
    music = inverse(_power_share(X - S, S) * X)
    music = _fit(music, n, mixture.shape[1], sr=sr, what="the Wiener-masked music")
    return mixture - music, music


def _stft_pair(sr: int, n: int, *, window_s: float = WIENER_WINDOW_S):
    """``(spec, inverse, freqs)``: the STFT of an ``(n, ch)`` array and its
    inverse (back to ``n`` samples), on one grid, and the grid's bin frequencies."""
    signal = require_package("scipy.signal")
    nperseg = min(_stft_size(sr, window_s), max(1, n))

    def spec(a):
        return signal.stft(np.asarray(a, dtype=np.float64).T, fs=sr, nperseg=nperseg)[2]

    def inverse(z):
        return signal.istft(z, fs=sr, nperseg=nperseg)[1].T[:n]

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    return spec, inverse, freqs


def _power_share(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``|A|^2 / (|A|^2 + |B|^2)`` per bin, in [0, 1] (0 where both are silent)."""
    A2, B2 = np.abs(A) ** 2, np.abs(B) ** 2
    return A2 / np.maximum(A2 + B2, WIENER_EPS)


def _blockwise_stft_map(
    signals: Sequence[np.ndarray],
    transform: Callable[..., Sequence[np.ndarray]],
    sr: int,
    *,
    window_s: float = WIENER_WINDOW_S,
    block_frames: int = STFT_BLOCK_FRAMES,
) -> list[np.ndarray]:
    """``istft(transform(freqs, *map(stft, signals)))``, one block of frames at a time.

    ``signals`` are 1-D, of one length ``n`` at least one STFT window long (the
    grid of :func:`_stft_pair`, then a power of two); ``transform`` maps their
    spectra, ``(bins, frames)`` each, to output spectra of that shape, bin and
    frame by bin and frame. The result equals
    ``inverse(transform(freqs, *map(spec, signals)))`` with :func:`_stft_pair`'s
    ``spec`` and ``inverse`` (scipy's ``stft``/``istft``, Hann, half overlap,
    zero boundary and padding): each block's spectra are scipy's own for those
    frames, and the overlap-add is scipy's, in the same order. Only a block of
    spectra is held at a time, never a whole signal's.

    >>> import numpy as np
    >>> a = np.random.default_rng(0).normal(size=5000)
    >>> spec, inverse, freqs = _stft_pair(8000, 5000)
    >>> want = inverse(0.5 * spec(a))
    >>> got, = _blockwise_stft_map([a], lambda f, A: [0.5 * A], 8000, block_frames=7)
    >>> bool(np.allclose(got, want, rtol=0, atol=1e-12))
    True
    """
    signal = require_package("scipy.signal")
    fft = require_package("scipy.fft")
    if block_frames < 1:
        raise ValueError(f"block_frames must be at least 1, got {block_frames}")
    n = len(signals[0])
    nperseg = _stft_size(sr, window_s)
    if n < nperseg or nperseg % 2:
        raise ValueError(
            f"a blockwise STFT needs at least one whole (even) window: "
            f"{n} samples for a {nperseg}-sample window"
        )
    half = nperseg // 2
    nstep = nperseg - half  # scipy's default noverlap = nperseg // 2
    extended = n + 2 * half  # boundary="zeros"
    tail = (-(extended - nperseg) % nstep) % nperseg  # padded=True
    total = extended + tail
    n_frames = (total - nperseg) // nstep + 1
    win = signal.get_window("hann", nperseg)
    win_sum, win2 = win.sum(), win**2
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)

    inputs = [np.asarray(a, dtype=np.float64) for a in signals]
    for a in inputs:
        if a.shape != (n,):
            raise ValueError(f"signals must all be 1-D of length {n}, got {a.shape}")

    def span(a, lo, hi):
        """Samples ``[lo, hi)`` of ``a`` zero-extended as scipy pads it (``half``
        zeros before, ``half + tail`` after): one block's worth, never a copy
        of the whole signal."""
        start, stop = lo - half, hi - half
        inside = a[max(0, start) : min(n, stop)]
        return np.pad(inside, (max(0, -start), max(0, stop - n)))

    outputs: Optional[list[np.ndarray]] = None
    norm = np.zeros(total)
    for f0 in range(0, n_frames, block_frames):
        f1 = min(n_frames, f0 + block_frames)
        lo, hi = f0 * nstep, (f1 - 1) * nstep + nperseg
        spectra = [
            signal.stft(
                span(a, lo, hi), fs=sr, nperseg=nperseg, boundary=None, padded=False
            )[2]
            for a in inputs
        ]
        results = transform(freqs, *spectra)
        del spectra
        if outputs is None:
            outputs = [np.zeros(total) for _ in results]
        for out, Z in zip(outputs, results):
            xsubs = fft.irfft(Z, axis=-2, n=nperseg)[:nperseg, :]
            xsubs *= win_sum  # scaling="spectrum", as istft does
            for k in range(f1 - f0):
                start = lo + k * nstep
                out[start : start + nperseg] += xsubs[:, k] * win
        del results
        for k in range(f0, f1):
            norm[k * nstep : k * nstep + nperseg] += win2
    divisor = norm[half : half + n]
    divisor[divisor <= 1e-10] = 1.0  # as istft: divide only where non-tiny
    results = []
    for out in outputs or []:
        kept = out[half : half + n]
        kept /= divisor
        results.append(kept)
    return results


def _split_round_trip(
    x: np.ndarray,
    speech: np.ndarray,
    music: np.ndarray,
    lost: np.ndarray,
    sr: int,
    *,
    heard_hz: Optional[float],
    window_s: float = WIENER_WINDOW_S,
    block_frames: int = STFT_BLOCK_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """A model's stems, resampled back to ``x``'s rate ``sr``, with what the
    resampling round trip ``lost`` (``x - back(x_native)``) put back.

    Below ``heard_hz`` (the model's Nyquist when it is below the input's: what
    it heard; ``None`` when it heard every band of ``x``) each stem gets its share of ``lost`` by the stems' power ratio, so
    what the resampler's slope weakened goes back to the stem it came from. At
    and above it the speech is silent and the music is ``x``: the model never
    heard that band (talk there included), and the stems hold only the images
    resampling made of them. So ``speech + music == x`` whenever the model's own
    stems sum to its input; a model that leaves something out (a dropped effects
    stem) still leaves it out. The line is as sharp as one STFT bin: content
    within about a bin of it is split between the stems.

    The STFTs are walked ``block_frames`` frames at a time
    (:func:`_blockwise_stft_map`), so the peak memory is a few blocks' spectra,
    not a few whole channels' (reeleehq/mixing#203); a signal shorter than one
    window takes the whole-signal path, which is small anyway.
    """
    n, ch = x.shape
    everything_heard = heard_hz is None
    if n >= _stft_size(sr, window_s):
        out_speech, out_music = np.empty_like(x), np.empty_like(x)

        def split(freqs, S, M, R, *X):
            heard = (freqs < (np.inf if everything_heard else heard_hz))[:, None]
            share = _power_share(S, M)
            S_out = np.where(heard, S + share * R, 0)
            M_out = M + (1 - share) * R
            if X:  # a band the model never heard: the music is x there
                M_out = np.where(heard, M_out, X[0])
            return S_out, M_out

        for c in range(ch):
            signals = [a[:, c] for a in (speech, music, lost)]
            if not everything_heard:
                signals.append(x[:, c])
            out_speech[:, c], out_music[:, c] = _blockwise_stft_map(
                signals, split, sr, window_s=window_s, block_frames=block_frames
            )
        return out_speech, out_music
    spec, inverse, freqs = _stft_pair(sr, n, window_s=window_s)
    heard = (freqs < (np.inf if everything_heard else heard_hz))[:, None]
    out_speech, out_music = np.empty_like(x), np.empty_like(x)
    for c in range(ch):
        S, M, R = (spec(a[:, c]) for a in (speech, music, lost))
        share = _power_share(S, M)
        S_out = np.where(heard, S + share * R, 0)
        M_out = M + (1 - share) * R
        del S, R, share
        if not everything_heard:
            M_out = np.where(heard, M_out, spec(x[:, c]))
        del M
        out_speech[:, c] = _fit(
            inverse(S_out)[:, None], n, 1, sr=sr, what="the speech"
        )[:, 0]
        out_music[:, c] = _fit(inverse(M_out)[:, None], n, 1, sr=sr, what="the music")[
            :, 0
        ]
    return out_speech, out_music


def invert_enhancer(enhance: SpeechEnhancer) -> SpeechSeparator:
    """A separator from a speech enhancer: music = mixture minus its speech.

    An enhancer keeps the voice and drops "noise"; to it, music (singing
    included) is noise. Inverting it gives speech-agnostic talk removal with a
    permissively licensed model. The mixture is split with a Wiener mask built
    from the two estimates (:func:`wiener_split`), so a poor estimate can make
    the music quieter but never louder than the mixture.

    >>> import numpy as np
    >>> sep = invert_enhancer(lambda a, sr: 0 * a)
    >>> s, m = sep(np.ones((4000, 1)), 8000)
    >>> bool(np.allclose(m, 1)), bool(np.allclose(s, 0))
    (True, True)
    """

    def separate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(samples, dtype=np.float64)
        n, ch = samples.shape
        speech = _fit(enhance(samples, sr), n, ch, sr=sr, what="the speech estimate")
        return wiener_split(samples, speech, sr)

    separate.__name__ = f"inverted_{getattr(enhance, '__name__', 'enhancer')}"
    return separate


# ---------------------------------------------------------------------------
# Strategies (heavy imports inside)
# ---------------------------------------------------------------------------


def _require_clearvoice() -> None:
    """Raise an ``ImportError`` with the no-deps install line if clearvoice is
    missing; warn if it is not the release the adapter was written against."""
    if importlib.util.find_spec("clearvoice") is None:
        raise ImportError(
            "the 'mossformer2_se_48k' separator requires clearvoice. Install it "
            f"with {CLEARVOICE_INSTALL_HINT} (clearvoice's own pins would "
            "downgrade numpy below 2 and add a second cv2).",
            name="clearvoice",
        )
    version = importlib.metadata.version("clearvoice")
    if version != CLEARVOICE_VERSION:
        warnings.warn(
            f"the 'mossformer2_se_48k' adapter was written against clearvoice "
            f"{CLEARVOICE_VERSION}'s internals; found {version}",
            stacklevel=3,
        )


@lru_cache(maxsize=1)
def _mossformer2_se_48k_model():
    """ClearVoice's MossFormer2-SE-48K, loaded once, from the Hugging Face cache.

    ClearVoice's own loader downloads into ``checkpoints/`` under the *current
    directory* and only prints a warning (then runs with random weights) when the
    download fails; downloading through ``huggingface_hub`` first puts the
    checkpoint in the shared HF cache and raises on failure.
    """
    _require_clearvoice()
    import huggingface_hub

    with warnings.catch_warnings():  # clearvoice's __init__ silences all warnings
        from clearvoice.network_wrapper import network_wrapper
        from clearvoice.networks import CLS_MossFormer2_SE_48K

    checkpoint_dir = huggingface_hub.snapshot_download(repo_id=MOSSFORMER2_SE_48K_REPO)
    if not os.path.isfile(os.path.join(checkpoint_dir, "last_best_checkpoint")):
        raise RuntimeError(
            f"{MOSSFORMER2_SE_48K_REPO} downloaded to {checkpoint_dir} without its "
            "'last_best_checkpoint' file"
        )
    wrapper = network_wrapper()
    wrapper.model_name = MOSSFORMER2_SE_48K_NAME
    wrapper.load_args_se()  # the model's packaged inference config
    args = wrapper.args
    args.task, args.network = "speech_enhancement", MOSSFORMER2_SE_48K_NAME
    args.checkpoint_dir = checkpoint_dir
    return CLS_MossFormer2_SE_48K(args)


def _clearvoice_tail_pad(n: int, args) -> int:
    """Zeros to append so clearvoice's decode writes every input sample.

    Two gaps, both at the end of the input (clearvoice 0.1.2):

    - every decode drops the last ``n % win_inc`` samples (its STFT hop, 8 ms
      at 48 kHz), so the input is first padded to a whole number of hops;
    - above ``one_time_decode_length`` seconds it decodes in windows and leaves
      the last ``(window - stride) / 2`` samples of its buffer at zero, padding
      the input only when ``(n - window)`` is not a multiple of the stride, so
      for exactly those lengths the last ~0.5 s would come back silent.

    The one-shot limit is a whole number of hops, so the first pad never moves
    an input onto the windowed path.
    """
    hop_pad = (-n) % args.win_inc
    n += hop_pad
    rate = args.sampling_rate
    if n <= rate * args.one_time_decode_length:
        return hop_pad
    window = int(rate * args.decode_window)
    stride = int(window * CLEARVOICE_STRIDE_FRACTION)
    return hop_pad + ((window - stride) // 2 if (n - window) % stride == 0 else 0)


def mossformer2_se_48k_speech(samples: np.ndarray, sr: int) -> np.ndarray:
    """MossFormer2-SE-48K's speech estimate of mono ``samples`` ``(n, 1)`` at 48 kHz."""
    _require_clearvoice()
    import torch

    with warnings.catch_warnings():
        from clearvoice.utils.decode import decode_one_audio

    net = _mossformer2_se_48k_model()
    mono = np.asarray(samples, dtype=np.float32)[:, 0]
    n = len(mono)
    mono = np.pad(mono, (0, _clearvoice_tail_pad(n, net.args)))
    with torch.no_grad():
        speech = decode_one_audio(net.model, net.device, mono[None, :], net.args)
    # The windowed decode (> one_time_decode_length) returns its zero-padded
    # working buffer, longer than the input: keep the input's span.
    return np.asarray(speech, dtype=np.float64)[:n, None]


@lru_cache(maxsize=2)
def _demucs_model(name: str):
    """A pretrained Demucs model (bag of models), loaded once per process."""
    from demucs.pretrained import get_model

    return get_model(name)


def htdemucs_separate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """htdemucs' ``vocals`` stem as speech, mixture minus it as music (44.1 kHz).

    For instrumental beds only: the vocals stem holds singing too. Mono input is
    duplicated to the model's two channels and averaged back.
    """
    import torch
    from demucs.apply import apply_model

    model = _demucs_model(HTDEMUCS_MODEL)
    x = np.asarray(samples, dtype=np.float64)
    n, ch = x.shape
    want = model.audio_channels
    if ch not in (1, want):
        raise ValueError(f"htdemucs takes 1 or {want} channels, got {ch}")
    wav = torch.from_numpy(np.tile(x, (1, want // ch)).T.astype(np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + HTDEMUCS_STD_EPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        stems = apply_model(model, ((wav - mean) / std)[None], device=device)[0]
    vocals = stems[model.sources.index(HTDEMUCS_SPEECH_STEM)] * std + mean
    vocals = vocals.cpu().numpy().T.astype(np.float64)
    if ch == 1:
        vocals = vocals.mean(axis=1, keepdims=True)
    return vocals, x - vocals


#: The registry name of :func:`voxy_separator`'s default (voxy's default model).
VOXY_SEPARATOR = "voxy_htdemucs_ft"


def voxy_separator(*, name: str = VOXY_SEPARATOR, **options) -> SeparationStrategy:
    """voxy's Demucs as a strategy, run **out of process** in voxy's own environment.

    ``voxy.separate.speech_separator(**options)`` (thorwhalen/voxy#99) behind the
    seam: ``htdemucs_ft``'s vocals sub-model by default, the music being the input
    minus the voice, at Demucs's 44.1 kHz (mixing resamples to and from it, as for
    ``"htdemucs"``). The host process imports no torch (extra ``voxy``,
    ``pip install 'mixing[voxy]'``); voxy builds its Demucs environment on first
    use. ``options`` are ``speech_separator``'s (``stem``, Demucs's ``shifts``,
    ``overlap``, a ``run`` such as another model's worker: give it its ``name``
    then), except ``sample_rate``, which is mixing's. Like ``"htdemucs"``, for
    instrumental beds: the vocals stem holds singing too.

    >>> voxy_separator().name, voxy_separator(shifts=2).context_s
    ('voxy_htdemucs_ft', 3.0)
    """
    if "sample_rate" in options:
        raise ValueError(
            "voxy_separator takes no sample_rate: mixing resamples to and from "
            "the model's rate itself"
        )

    def separate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        try:  # the extra is checked first; a voxy too old has no speech_separator
            from voxy.separate import speech_separator
        except ImportError as e:
            raise ImportError(
                f"the {name!r} separator needs a voxy with speech_separator: "
                f"{install_hint(VOXY_EXTRA)} (upgrades voxy to the floor)",
                name=e.name,
            ) from e
        return speech_separator(**options)(samples, sr)

    return SeparationStrategy(
        name=name,
        separate=separate,
        license="MIT",
        commercial_ok=True,
        extra=VOXY_EXTRA,
        mono=False,
        sample_rate=HTDEMUCS_RATE,  # Demucs's: mixing resamples around it, as for htdemucs
        context_s=HTDEMUCS_CONTEXT_S,
    )


#: The registry: ``{name: SeparationStrategy}``. Add an entry to register a
#: strategy; filter with e.g. ``{k: s for k, s in SEPARATORS.items() if
#: s.commercial_ok}``. Not registered on purpose: Bandit-v2 (CC-BY-SA weights,
#: inference in a non-pip repo: pass it as a callable) and CC BY-NC checkpoints
#: (BandIt Plus, dns64). #63's scores (SI-SDR of the music, talk over singing /
#: over an instrumental): mossformer2_se_48k +7.3 / +0.8 dB, htdemucs -3.7 / +5.4.
SEPARATORS: dict[str, SeparationStrategy] = {
    "mossformer2_se_48k": SeparationStrategy(
        name="mossformer2_se_48k",
        separate=invert_enhancer(mossformer2_se_48k_speech),
        license="Apache-2.0",
        commercial_ok=True,
        extra=SEPARATION_EXTRA,
        mono=True,
        sample_rate=MOSSFORMER2_SE_48K_RATE,
        context_s=MOSSFORMER2_SE_48K_CONTEXT_S,
    ),
    "htdemucs": SeparationStrategy(
        name="htdemucs",
        separate=htdemucs_separate,
        license="MIT",
        commercial_ok=True,
        extra=DEMUCS_EXTRA,
        mono=False,
        sample_rate=HTDEMUCS_RATE,
        context_s=HTDEMUCS_CONTEXT_S,
    ),
    VOXY_SEPARATOR: voxy_separator(),
}
#: The default strategy: the strongest permissive, pip-installable one (#63, #120).
DEFAULT_SEPARATOR = "mossformer2_se_48k"


def separate_speech(
    audio: AudioLike,
    *,
    separator: Union[str, SeparationStrategy, SpeechSeparator] = DEFAULT_SEPARATOR,
    sr: Optional[int] = None,
):
    """Split ``audio`` into ``(speech, music)``.

    Args:
        audio: an :class:`Audio`, a path, an ``AudioSamples`` view, or samples
            ``(n,)`` / ``(n, channels)`` with ``sr`` (float in [-1, 1], or
            signed-integer PCM, scaled by its full scale).
        separator: a key of :data:`SEPARATORS` (default
            :data:`DEFAULT_SEPARATOR`), a :class:`SeparationStrategy`, or a
            callable ``(samples (n, ch), sr) -> (speech, music)``.
        sr: sample rate of an array ``audio`` (ignored otherwise, but must match).

    Returns:
        ``(speech, music)`` of the input's kind: two ``Audio`` s like it, or two
        arrays of its shape and dtype (integer PCM and ``Audio`` saturate beyond
        full scale).

    >>> import numpy as np
    >>> s, m = separate_speech(np.ones(4), separator=lambda a, r: (a * 0, a), sr=8000)
    >>> m.tolist()
    [1.0, 1.0, 1.0, 1.0]
    """
    strategy = _resolve(separator)
    source = float_input(audio, sr)
    speech, music = strategy(source.samples, source.sr)
    return source.restore(speech), source.restore(music)


# ---------------------------------------------------------------------------
# remove_talk: separation inside spans, spliced with linear fades
# ---------------------------------------------------------------------------


def _splice_plan(spans: list[Span], *, margin: float, sr: int, n: int):
    """``[(a, b, weights)]``: frame runs ``[a, b)`` to replace (spans widened by
    ``margin``, clamped to ``[0, n]``, overlapping ones merged) and the weight of
    the music estimate on each frame: 1 inside a span, linear ramps over the
    margin, the max where widened spans overlap; ``margin=0`` is a hard cut."""
    m = int(round(margin * sr))
    cores = sorted((int(round(s * sr)), int(round(e * sr))) for s, e in spans)
    groups: list[list] = []
    for start, end in cores:
        a, b = max(0, start - m), min(n, end + m)
        if b <= a:
            continue
        if groups and a <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], b)
            groups[-1][2].append((start, end))
        else:
            groups.append([a, b, [(start, end)]])

    def weights(a, b, members):
        centres = np.arange(a, b) + 0.5
        w = np.zeros(b - a)
        for start, end in members:
            if m == 0:
                inside = (centres > start) & (centres < end)
                w = np.maximum(w, inside.astype(float))
            else:
                ramp = np.minimum(centres - (start - m), (end + m) - centres) / m
                w = np.maximum(w, np.clip(ramp, 0.0, 1.0))
        return w

    return [(a, b, weights(a, b, members)) for a, b, members in groups]


def remove_talk(
    audio: AudioLike,
    spans_or_edges,
    *,
    separator: Union[str, SeparationStrategy, SpeechSeparator] = DEFAULT_SEPARATOR,
    margin: float = DEFAULT_MARGIN_S,
    sr: Optional[int] = None,
    output: Output = None,
    **save_kwargs,
):
    """``audio`` with the talk inside ``spans_or_edges`` removed, the music kept.

    Each span is widened by ``margin`` on both sides (clamped to the file;
    overlapping widened spans merge). The separator runs on each widened span plus
    the strategy's ``context_s`` (not the whole file); its music estimate fades
    in linearly over the margin before the span, replaces the input inside it,
    and fades out over the margin after it (``margin=0``: a hard cut). Everything
    outside the widened spans is sample-identical to the input.

    Args:
        audio: an :class:`Audio`, a path, an ``AudioSamples`` view, or samples
            with ``sr`` (see :func:`separate_speech`).
        spans_or_edges: where the talk is, in seconds (whatever the
            ``Audio.time_unit``): a
            :class:`~mixing.audio.talk_edges.TalkEdges` (its intro and outro), a
            ``Segment``, a ``(start, end)`` pair, a ``(2,)`` / ``(k, 2)`` array,
            or an iterable of these. Each needs finite ``end > start``.
        separator: see :func:`separate_speech`.
        margin: seconds added to each side of every span, faded over.
        sr: sample rate of an array ``audio``.
        output: None (return the result), a file path, a directory (writes
            ``<input stem>_talk_removed.wav``), or a callable sink; see
            :mod:`mixing.egress`.
        **save_kwargs: passed to ``Audio.save`` when ``output`` is a path (array
            results are written as 32-bit PCM through ``Audio`` too).

    Returns:
        The input's kind (an ``Audio`` like it, or an array of its shape and
        dtype), or what ``output`` makes of it.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    source = float_input(audio, sr)
    x, rate = source.samples, source.sr
    plan = _splice_plan(as_spans(spans_or_edges), margin=margin, sr=rate, n=len(x))
    out = x.copy()
    if plan:
        strategy = _resolve(separator)
        context = int(round(strategy.context_s * rate))
        for a, b, w in plan:
            lo, hi = max(0, a - context), min(len(x), b + context)
            music = strategy(x[lo:hi], rate)[1][a - lo : b - lo]
            out[a:b] = x[a:b] + w[:, None] * (music - x[a:b])
    return deliver(
        source.restore(out),
        output,
        write=lambda _, path: source.as_audio(out).save(path, **save_kwargs),
        default_name=f"{source.stem}{OUTPUT_SUFFIX}",
    )


__all__ = [
    "SpeechSeparator",
    "SpeechEnhancer",
    "SeparationStrategy",
    "SEPARATORS",
    "DEFAULT_SEPARATOR",
    "DEFAULT_MARGIN_S",
    "separate_speech",
    "remove_talk",
    "invert_enhancer",
    "wiener_split",
    "mossformer2_se_48k_speech",
    "htdemucs_separate",
    "voxy_separator",
    "VOXY_SEPARATOR",
]
