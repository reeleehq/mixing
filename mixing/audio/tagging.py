"""Per-window speech / music probabilities from a pretrained audio tagger.

This is the *classifier seam* behind the ``"audioset"`` segmentation strategy
(:func:`mixing.audio.segmentation.segment_by_audioset`). A classifier is any
callable ``(samples, sr) -> WindowProbabilities``: it slides a window over a mono
float signal and says, for each window, how likely it holds speech and how likely
it holds music. The strategy only consumes that contract, so any tagger can be
injected (foley's ``PannsTagger`` or a CLAP tagger, wrapped in a few lines)
without ``mixing`` depending on it.

The default, :func:`ast_audioset_classifier`, is the Audio Spectrogram Transformer
fine-tuned on AudioSet (``MIT/ast-finetuned-audioset-10-10-0.4593``, via
``transformers``). Speech is the max over the AudioSet classes *Speech*,
*Narration, monologue* and *Conversation*; music is the max over *Music* and
*Musical instrument*. Unlike handcrafted features (low-energy ratio, 4 Hz
modulation), it tells sung vocals from talk.

Heavy dependencies (``torch``, ``transformers``) come with the ``mixing[ml]``
extra and are imported lazily, inside the classifier, so ``import mixing.audio``
stays light. The model is loaded once per process (cached).

>>> import numpy as np
>>> probs = WindowProbabilities(
...     starts=np.array([0.0, 2.0]), ends=np.array([5.0, 7.0]),
...     speech=np.array([0.9, 0.1]), music=np.array([0.1, 0.8]),
... )
>>> len(probs)
2
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from ..util import require_ml_extra, require_package

#: The default AudioSet tagger (Audio Spectrogram Transformer, MIT, ~87M params).
DEFAULT_AUDIOSET_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
#: Sample rate the AST feature extractor expects.
AST_SAMPLE_RATE = 16000
#: Analysis window length (seconds) fed to the tagger.
DEFAULT_WINDOW_SECONDS = 5.0
#: Hop between consecutive windows (seconds).
DEFAULT_HOP_SECONDS = 2.0
#: Windows per forward pass.
DEFAULT_BATCH_SIZE = 16
#: AudioSet class names whose max is the "speech" probability.
SPEECH_CLASSES = ("Speech", "Narration, monologue", "Conversation")
#: AudioSet class names whose max is the "music" probability.
MUSIC_CLASSES = ("Music", "Musical instrument")


@dataclass(frozen=True)
class WindowProbabilities:
    """Speech and music probabilities for a sequence of (possibly overlapping) windows.

    Attributes:
        starts: Window start times in seconds, ascending.
        ends: Window end times in seconds (``ends[i] > starts[i]``).
        speech: Probability, in [0, 1], that window ``i`` holds speech.
        music: Probability, in [0, 1], that window ``i`` holds music.
    """

    starts: np.ndarray
    ends: np.ndarray
    speech: np.ndarray
    music: np.ndarray

    def __post_init__(self):
        n = len(self.starts)
        if not (len(self.ends) == len(self.speech) == len(self.music) == n):
            raise ValueError(
                "WindowProbabilities arrays must all have the same length; got "
                f"starts={n}, ends={len(self.ends)}, speech={len(self.speech)}, "
                f"music={len(self.music)}."
            )
        starts, ends = np.asarray(self.starts), np.asarray(self.ends)
        if n and (np.any(ends <= starts) or np.any(np.diff(starts) < 0)):
            raise ValueError(
                "WindowProbabilities needs ascending starts (seconds) and "
                "ends > starts for every window."
            )

    def __len__(self) -> int:
        return len(self.starts)


#: The classifier seam: ``(mono float samples, sample rate) -> WindowProbabilities``.
AudioClassifier = Callable[[np.ndarray, int], WindowProbabilities]


def window_starts(
    n_samples: int, sr: int, *, window_seconds: float, hop_seconds: float
) -> np.ndarray:
    """Start sample index of every analysis window (at least one, even if short).

    >>> window_starts(16000 * 12, 16000, window_seconds=5, hop_seconds=2).tolist()
    [0, 32000, 64000, 96000]
    """
    win, hop = int(window_seconds * sr), max(1, int(hop_seconds * sr))
    return np.arange(0, max(1, n_samples - win + 1), hop)


def _pick_device(device: str | None) -> str:
    """Resolve ``device=None`` to the best available torch device (mps > cuda > cpu)."""
    if device is not None:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=4)
def _load_ast(model_name: str, device: str):
    """Load (once per process) the feature extractor and model, on ``device``."""
    require_ml_extra("the 'audioset' classifier")
    import transformers

    extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    model = transformers.ASTForAudioClassification.from_pretrained(model_name).eval()
    return extractor, model.to(device)


def _class_indices(label2id: dict, names: Sequence[str]) -> list[int]:
    missing = [n for n in names if n not in label2id]
    if missing:
        raise ValueError(f"Model has no AudioSet classes named {missing}.")
    return [label2id[n] for n in names]


def ast_audioset_classifier(
    *,
    model_name: str = DEFAULT_AUDIOSET_MODEL,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    speech_classes: Sequence[str] = SPEECH_CLASSES,
    music_classes: Sequence[str] = MUSIC_CLASSES,
) -> AudioClassifier:
    """Make the default classifier: an AudioSet AST tagger over sliding windows.

    Nothing heavy happens here; the model is loaded on the first call of the
    returned classifier (and cached per process).

    Args:
        model_name: Hugging Face id of an AST AudioSet checkpoint.
        window_seconds: Window length fed to the model.
        hop_seconds: Hop between windows. Smaller = finer boundaries, slower.
        batch_size: Windows per forward pass.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``; ``None`` picks the best available.
        speech_classes: AudioSet classes whose max is the speech probability.
        music_classes: AudioSet classes whose max is the music probability.

    Returns:
        A callable ``(samples, sr) -> WindowProbabilities``. Input is resampled to
        16 kHz if needed.
    """

    def classify(samples: np.ndarray, sr: int) -> WindowProbabilities:
        require_ml_extra("the 'audioset' classifier")
        import torch

        x = np.asarray(samples, dtype=np.float32)
        if sr != AST_SAMPLE_RATE:
            resample_poly = require_package("scipy.signal").resample_poly
            g = np.gcd(int(sr), AST_SAMPLE_RATE)
            x = resample_poly(x, AST_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
        dev = _pick_device(device)
        extractor, model = _load_ast(model_name, dev)
        sp_idx = _class_indices(model.config.label2id, speech_classes)
        mu_idx = _class_indices(model.config.label2id, music_classes)
        win = int(window_seconds * AST_SAMPLE_RATE)
        starts = window_starts(
            len(x),
            AST_SAMPLE_RATE,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
        speech, music = [], []
        for b in range(0, len(starts), batch_size):
            chunk = [x[s : s + win] for s in starts[b : b + batch_size]]
            inputs = extractor(
                chunk, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
            ).to(dev)
            with torch.no_grad():
                p = torch.sigmoid(model(**inputs).logits).float().cpu().numpy()
            speech.append(p[:, sp_idx].max(axis=1))
            music.append(p[:, mu_idx].max(axis=1))
        t0 = starts / AST_SAMPLE_RATE
        return WindowProbabilities(
            starts=t0,
            ends=np.minimum(t0 + window_seconds, len(x) / AST_SAMPLE_RATE),
            speech=np.concatenate(speech),
            music=np.concatenate(music),
        )

    return classify
