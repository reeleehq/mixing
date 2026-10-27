"""Separate speech from music, and remove a host's talk while keeping the music.

Two layers, each usable on its own:

- :func:`separate_speech` ``(audio) -> (speech, music)``: the **separation
  seam**. ``separator=`` is a key of :data:`SEPARATORS`, a
  :class:`SeparationStrategy`, or any callable ``(samples (n, ch), sr) ->
  (speech, music)``, so a model mixing does not ship (Bandit-v2 from ZFTurbo's
  MSST repo, a host-fine-tuned network) plugs in without mixing depending on it.
- :func:`remove_talk` ``(audio, spans) -> audio``: the **splice combinator**. It
  runs the separator only on the given talk spans (widened by ``margin``, plus
  model context), crossfades the music estimate in and out linearly, and leaves
  every sample outside the widened spans identical to the input. The spans can
  come from anywhere: :class:`~mixing.audio.talk_edges.TalkEdges` (from
  ``find_talk_edges``), ``Segment`` s, or plain ``(start, end)`` pairs in seconds.

A :class:`SeparationStrategy` wraps a model with what the framework needs to run
it: its native sample rate (audio is resampled in and out; when the model's rate
is lower than the input's, the band above the model's Nyquist frequency, which
the model never saw, goes to the music untouched), whether it is mono-only (then
it runs once per channel: radio music is wide and the talk sits near the centre,
so per-channel runs matter), how much context it needs around a span, the mixing
extra that installs it, and its licence (``commercial_ok`` lets a caller filter).

Registered strategies (licences are the checkpoints'):

- ``"mossformer2_se_48k"`` (default; extra ``separation``, pip ``clearvoice``,
  Apache-2.0): the MossFormer2-SE-48K speech enhancer, *inverted* by
  :func:`invert_enhancer`: music is the mixture minus the speech estimate, with a
  Wiener mask. Keeps singing (it is not speech to an enhancer).
- ``"htdemucs"`` (extra ``demucs``, pip ``demucs``, MIT): the ``vocals`` stem of
  Hybrid Transformer Demucs as the speech. **Instrumental beds only**: it removes
  the singing along with the talk.

Heavy dependencies (``torch``, ``clearvoice``, ``demucs``) are imported inside the
strategies, so ``import mixing.audio`` stays light, and a missing one raises an
``ImportError`` naming the extra. Models load once per process.

>>> import numpy as np
>>> sr = 8000
>>> x = np.full((sr, 1), 0.5)                       # 1 s of a constant "mixture"
>>> halve = lambda a, rate: (a / 2, a / 2)          # a toy separator
>>> out = remove_talk(x, [(0.4, 0.6)], separator=halve, sr=sr, margin=0.1, crossfade=0.05)
>>> float(out[0, 0]), float(out[4000, 0]), float(out[-1, 0])
(0.5, 0.25, 0.5)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

from ..egress import Output, deliver
from ..util import DEMUCS_EXTRA, SEPARATION_EXTRA, require_extra, require_package
from ._pcm import resample, segment_samples
from .audio_ops import Audio
from .samples import to_audio

#: The separator seam: ``(float samples (n, channels), sr) -> (speech, music)``,
#: both shaped like the input and at the same rate.
SpeechSeparator = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
#: A speech enhancer: ``(float samples (n, channels), sr) -> speech estimate``.
SpeechEnhancer = Callable[[np.ndarray, int], np.ndarray]
#: A talk span in seconds.
Span = tuple[float, float]

#: Each talk span is widened by this much (seconds) on both sides: it absorbs the
#: timing error of the span's producer (``find_talk_edges`` without a transcript
#: is window-resolution) and the soft onset and decay of a voice.
DEFAULT_MARGIN_S = 0.3
#: Length (seconds) of the linear crossfades at each widened span's edges, inside
#: it. Equal to the margin, the crossfade ends exactly where the given span starts,
#: so the span itself is all music estimate (the splice #63's experiment scored).
DEFAULT_CROSSFADE_S = 0.3
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
#: zero (they get mask 0.5, which is harmless: both sources are silent there).
WIENER_EPS = 1e-20

#: Hugging Face repo of the MossFormer2-SE-48K checkpoint (Apache-2.0, ~220 MB).
MOSSFORMER2_SE_48K_REPO = "alibabasglab/MossFormer2_SE_48K"
#: The model name ClearVoice (clearvoice) knows it by.
MOSSFORMER2_SE_48K_NAME = "MossFormer2_SE_48K"
#: Native sample rate of MossFormer2-SE-48K.
MOSSFORMER2_SE_48K_RATE = 48000
#: Context (seconds) around a span for MossFormer2-SE: its STFT frames are 40 ms
#: and its sequence model sees the whole input, so a second each side suffices.
MOSSFORMER2_SE_48K_CONTEXT_S = 1.0
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
        name: Registry key / display name.
        separate: ``(samples (n, ch), sr) -> (speech, music)`` at ``sample_rate``.
        license: Licence of the weights (and code) the strategy runs.
        commercial_ok: Whether that licence allows commercial use (``None``:
            unknown, as for a caller's bare callable).
        extra: The mixing extra that installs the strategy's dependencies, or
            ``None``; checked before the strategy runs.
        mono: True if the model takes one channel: it then runs per channel.
        sample_rate: The model's native rate, or ``None`` for any rate.
        context_s: Seconds of audio the model wants around a span
            (:func:`remove_talk` cuts this much context on each side).
        notes: When to use it (and when not).
    """

    name: str
    separate: SpeechSeparator = field(repr=False)
    license: Optional[str] = None
    commercial_ok: Optional[bool] = None
    extra: Optional[str] = None
    mono: bool = False
    sample_rate: Optional[int] = None
    context_s: float = DEFAULT_CONTEXT_S
    notes: str = ""

    def __call__(self, samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        return _run_strategy(self, samples, sr)


# ---------------------------------------------------------------------------
# The framework: resampling, per-channel runs, length fitting
# ---------------------------------------------------------------------------


def _fit(out: Any, n: int, channels: int, *, sr: int, who: str) -> np.ndarray:
    """``out`` as float ``(n, channels)``: 1-D is one channel, the length is padded
    or cut to ``n`` within :data:`LENGTH_TOLERANCE_S`."""
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[1] != channels:
        raise ValueError(
            f"separator {who!r} must return arrays of shape (n, {channels}), "
            f"got {out.shape}"
        )
    if abs(len(out) - n) > max(1, int(round(LENGTH_TOLERANCE_S * sr))):
        raise ValueError(
            f"separator {who!r} returned {len(out)} frames for {n} "
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
            _fit(speech, n, width, sr=sr, who=strategy.name),
            _fit(music, n, width, sr=sr, who=strategy.name),
        )

    if not strategy.mono or ch == 1:
        return run(x)
    pairs = [run(x[:, [c]]) for c in range(ch)]
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
    back = lambda y: _fit(resample(y, native, sr), n, ch, sr=sr, who=strategy.name)
    speech, music = back(speech), back(music)
    if native < sr:  # the band the model never saw passes through, as music
        music = music + (x - back(x_native))
    return speech, music


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
    ``|M|^2 / (|M|^2 + |S|^2)``). The two outputs sum to the mixture. Soft masking
    is more forgiving than subtracting waveforms: an estimate that is slightly off
    in phase or level does not leave a ghost of the voice in the music.

    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(4000, 1))
    >>> s, m = wiener_split(x, 0.5 * x, 8000)
    >>> bool(np.allclose(s + m, x))
    True
    """
    signal = require_package("scipy.signal")
    stft, istft = signal.stft, signal.istft
    mixture = np.asarray(mixture, dtype=np.float64)
    n = len(mixture)
    nperseg = _stft_size(sr, window_s)
    if n < nperseg:  # too short to mask: the plain residual
        speech = np.asarray(speech, dtype=np.float64)
        return speech, mixture - speech
    spec = lambda y: stft(np.asarray(y, dtype=np.float64).T, nperseg=nperseg)[2]
    X, S = spec(mixture), spec(speech)
    M2, S2 = np.abs(X - S) ** 2, np.abs(S) ** 2
    mask = M2 / np.maximum(M2 + S2, WIENER_EPS)
    music = istft(mask * X, nperseg=nperseg)[1].T[:n]
    music = _fit(music, n, mixture.shape[1], sr=sr, who="wiener_split")
    return mixture - music, music


def invert_enhancer(
    enhance: SpeechEnhancer, *, wiener: bool = True, window_s: float = WIENER_WINDOW_S
) -> SpeechSeparator:
    """A separator from a speech enhancer: music = mixture minus its speech.

    An enhancer keeps the voice and drops "noise"; to it, music (singing
    included) is noise. Inverting it gives speech-agnostic talk removal with a
    permissively licensed model. ``wiener=True`` splits the mixture with a Wiener
    mask built from the two estimates (:func:`wiener_split`); ``False`` subtracts
    the waveforms.

    >>> import numpy as np
    >>> sep = invert_enhancer(lambda a, sr: 0.25 * a, wiener=False)
    >>> s, m = sep(np.ones((10, 1)), 8000)
    >>> float(s[0, 0]), float(m[0, 0])
    (0.25, 0.75)
    """

    def separate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(samples, dtype=np.float64)
        n, ch = samples.shape
        speech = _fit(enhance(samples, sr), n, ch, sr=sr, who="enhancer")
        if wiener:
            return wiener_split(samples, speech, sr, window_s=window_s)
        return speech, samples - speech

    separate.__name__ = f"inverted_{getattr(enhance, '__name__', 'enhancer')}"
    return separate


# ---------------------------------------------------------------------------
# Strategies (heavy imports inside)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _mossformer2_se_48k_model():
    """ClearVoice's MossFormer2-SE-48K, loaded once, from the Hugging Face cache.

    ClearVoice's own loader downloads into ``checkpoints/`` under the *current
    directory* and only prints a warning (then runs with random weights) when the
    download fails; downloading through ``huggingface_hub`` first puts the
    checkpoint in the shared HF cache and raises on failure.
    """
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


def mossformer2_se_48k_speech(samples: np.ndarray, sr: int) -> np.ndarray:
    """MossFormer2-SE-48K's speech estimate of mono ``samples`` ``(n, 1)`` at 48 kHz."""
    import torch

    with warnings.catch_warnings():
        from clearvoice.utils.decode import decode_one_audio

    net = _mossformer2_se_48k_model()
    mono = np.asarray(samples, dtype=np.float32)[:, 0][None, :]
    with torch.no_grad():
        speech = decode_one_audio(net.model, net.device, mono, net.args)
    return np.asarray(speech, dtype=np.float64)[:, None]


@lru_cache(maxsize=2)
def _demucs_model(name: str):
    """A pretrained Demucs model (bag of models), loaded once per process."""
    from demucs.pretrained import get_model

    return get_model(name)


def htdemucs_separate(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """htdemucs' ``vocals`` stem as speech, mixture minus it as music (44.1 kHz).

    Mono input is duplicated to the model's two channels and averaged back.
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


#: The registry: ``{name: SeparationStrategy}``. Add an entry to register a
#: strategy; filter with e.g. ``{k: s for k, s in SEPARATORS.items() if
#: s.commercial_ok}``. Not registered on purpose: Bandit-v2 (CC-BY-SA weights,
#: inference in a non-pip repo: pass it as a callable) and CC BY-NC checkpoints
#: (BandIt Plus, dns64).
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
        notes=(
            "MossFormer2-SE-48K speech enhancer, inverted (mixture minus speech, "
            "Wiener mask). Keeps singing. #63: +7.3 dB SI-SDR over singing, +0.8 "
            "over an instrumental."
        ),
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
        notes=(
            "Instrumental beds only: the 'vocals' stem is taken as speech, so "
            "singing goes too. #63: +5.4 dB on an instrumental, -3.7 over singing."
        ),
    ),
}
#: The default strategy: the strongest permissive, pip-installable one (#63, #120).
DEFAULT_SEPARATOR = "mossformer2_se_48k"


# ---------------------------------------------------------------------------
# Inputs: Audio, path, or array + sr
# ---------------------------------------------------------------------------

AudioLike = Union[str, Path, Audio, np.ndarray, Any]


def _as_samples(audio: AudioLike, sr: Optional[int]):
    """``(float samples (n, ch), sr, restore)``; ``restore(y)`` turns a float
    result back into the input's kind (an ``Audio`` like it, or an array)."""
    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError("an array input needs its sample rate: pass sr=")
        if audio.ndim not in (1, 2):
            raise ValueError(
                f"samples must be (n,) or (n, channels), got {audio.shape}"
            )
        x = np.asarray(audio, dtype=np.float64)
        one_d, dtype = audio.ndim == 1, audio.dtype

        def restore(y):
            y = y[:, 0] if one_d else y
            return y.astype(dtype) if np.issubdtype(dtype, np.floating) else y

        return (x[:, None] if one_d else x), int(sr), restore
    audio = audio if isinstance(audio, Audio) else Audio(audio)
    segment = audio._get_segment()
    if sr is not None and sr != segment.frame_rate:
        raise ValueError(f"sr={sr} given for audio at {segment.frame_rate} Hz")
    return (
        segment_samples(segment),
        segment.frame_rate,
        lambda y: to_audio(y, segment, time_unit=audio.time_unit),
    )


def separate_speech(
    audio: AudioLike,
    *,
    separator: Union[str, SeparationStrategy, SpeechSeparator] = DEFAULT_SEPARATOR,
    sr: Optional[int] = None,
):
    """Split ``audio`` into ``(speech, music)``.

    Args:
        audio: an :class:`Audio`, a path, or float samples ``(n,)`` /
            ``(n, channels)`` with ``sr``.
        separator: a key of :data:`SEPARATORS` (default
            :data:`DEFAULT_SEPARATOR`), a :class:`SeparationStrategy`, or a
            callable ``(samples (n, ch), sr) -> (speech, music)``.
        sr: sample rate of an array ``audio`` (ignored otherwise, but must match).

    Returns:
        ``(speech, music)`` of the input's kind: two ``Audio`` s like it (values
        beyond full scale saturate) or two arrays shaped like it.

    >>> import numpy as np
    >>> s, m = separate_speech(np.ones(4), separator=lambda a, r: (a * 0, a), sr=8000)
    >>> m.tolist()
    [1.0, 1.0, 1.0, 1.0]
    """
    strategy = _resolve(separator)
    x, rate, restore = _as_samples(audio, sr)
    speech, music = strategy(x, rate)
    return restore(speech), restore(music)


# ---------------------------------------------------------------------------
# remove_talk: separation inside spans, spliced with crossfades
# ---------------------------------------------------------------------------


def _is_number_pair(obj) -> bool:
    return (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and all(isinstance(v, Real) for v in obj)
    )


def _as_spans(spans_or_edges) -> list[Span]:
    """Talk spans in seconds from ``TalkEdges`` (its intro/outro), ``Segment`` s,
    ``(start, end)`` pairs, or an iterable of any of them."""
    obj = spans_or_edges
    if obj is None:
        return []
    if hasattr(obj, "intro") and hasattr(obj, "outro"):  # TalkEdges
        return [(s.start, s.end) for s in (obj.intro, obj.outro) if s is not None]
    if hasattr(obj, "start") and hasattr(obj, "end"):  # Segment-like
        return [(float(obj.start), float(obj.end))]
    if _is_number_pair(obj):
        start, end = float(obj[0]), float(obj[1])
        if not end > start:
            raise ValueError(f"a talk span needs end > start, got {(start, end)}")
        return [(start, end)]
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return [span for item in obj for span in _as_spans(item)]
    raise TypeError(
        f"talk spans must be TalkEdges, Segments or (start, end) pairs, got {obj!r}"
    )


def _frame_ranges(spans: list[Span], *, margin: float, sr: int, n: int):
    """Spans widened by ``margin``, in frames, clamped to ``[0, n]``, sorted, with
    overlapping or touching ones merged."""
    ranges = sorted(
        (max(0, int(round((s - margin) * sr))), min(n, int(round((e + margin) * sr))))
        for s, e in spans
    )
    merged: list[list[int]] = []
    for a, b in ranges:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(r) for r in merged]


def _splice_weights(a: int, b: int, n: int, xf: int) -> np.ndarray:
    """Weight of the separated music over frames ``[a, b)``: 1, with linear ramps
    of ``xf`` frames at an edge that is not the file's edge."""
    w = np.ones(b - a)
    xf = min(xf, (b - a) // 2)
    if xf > 0:
        ramp = (np.arange(xf) + 0.5) / xf
        if a > 0:
            w[:xf] = ramp
        if b < n:
            w[-xf:] = ramp[::-1]
    return w


def _write_array(samples: np.ndarray, path: Path, sr: int) -> None:
    """Float samples to a 32-bit float WAV."""
    wavfile = require_package("scipy.io.wavfile")
    wavfile.write(str(path), sr, np.asarray(samples, dtype=np.float32))


def remove_talk(
    audio: AudioLike,
    spans_or_edges,
    *,
    separator: Union[str, SeparationStrategy, SpeechSeparator] = DEFAULT_SEPARATOR,
    margin: float = DEFAULT_MARGIN_S,
    crossfade: float = DEFAULT_CROSSFADE_S,
    sr: Optional[int] = None,
    output: Output = None,
    **save_kwargs,
):
    """``audio`` with the talk inside ``spans_or_edges`` removed, the music kept.

    Each span is widened by ``margin`` on both sides (clamped to the file;
    overlapping widened spans merge), the separator runs on the widened span plus
    the strategy's ``context_s`` (not the whole file), and its music estimate
    replaces the input there, with linear crossfades of ``crossfade`` seconds just
    inside the widened span's edges (none at a file edge). Everything outside the
    widened spans is sample-identical to the input.

    Args:
        audio: an :class:`Audio`, a path, or float samples with ``sr``.
        spans_or_edges: where the talk is, in seconds (whatever the
            ``Audio.time_unit``): a
            :class:`~mixing.audio.talk_edges.TalkEdges` (its intro and outro), a
            ``Segment``, a ``(start, end)`` pair, or an iterable of these.
        separator: see :func:`separate_speech`.
        margin: seconds added to each side of every span.
        crossfade: length (seconds) of each crossfade; at most half a span.
        sr: sample rate of an array ``audio``.
        output: None (return the result), a file path, a directory, or a callable
            sink; see :mod:`mixing.egress`.
        **save_kwargs: passed to ``Audio.save`` when ``output`` is a path.

    Returns:
        The input's kind (an ``Audio`` like it, or an array shaped like it), or
        what ``output`` makes of it.
    """
    if margin < 0 or crossfade < 0:
        raise ValueError(
            f"margin and crossfade must be >= 0, got {margin}, {crossfade}"
        )
    x, rate, restore = _as_samples(audio, sr)
    n = len(x)
    ranges = _frame_ranges(_as_spans(spans_or_edges), margin=margin, sr=rate, n=n)
    out = x.copy()
    if ranges:
        strategy = _resolve(separator)
        context = int(round(strategy.context_s * rate))
        xf = int(round(crossfade * rate))
        for a, b in ranges:
            lo, hi = max(0, a - context), min(n, b + context)
            music = strategy(x[lo:hi], rate)[1][a - lo : b - lo]
            w = _splice_weights(a, b, n, xf)[:, None]
            out[a:b] = x[a:b] + w * (music - x[a:b])
    result = restore(out)
    if isinstance(result, Audio):
        write = lambda r, p: r.save(p, **save_kwargs)
    else:
        write = lambda r, p: _write_array(r, p, rate)
    return deliver(result, output, write=write, default_name="audio_talk_removed.wav")


__all__ = [
    "SpeechSeparator",
    "SpeechEnhancer",
    "SeparationStrategy",
    "SEPARATORS",
    "DEFAULT_SEPARATOR",
    "DEFAULT_MARGIN_S",
    "DEFAULT_CROSSFADE_S",
    "separate_speech",
    "remove_talk",
    "invert_enhancer",
    "wiener_split",
    "mossformer2_se_48k_speech",
    "htdemucs_separate",
]
