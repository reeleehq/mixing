"""Beat / downbeat / onset analysis of an audio signal — the ``mixing[beats]`` primitive.

A small, permissively-licensed (librosa, ISC) audio-analysis primitive that any consumer
can reuse to answer "where are the beats and where is the rhythmic energy in this audio".
Its first customer is muvid's footage-scoring layer (thorwhalen/muvid#13), which computes a
:func:`beat_grid` **once on the clean master song** and maps every clip onto it via the
clip's known offset — but nothing here is footage-specific.

Design notes:

- **Lazy heavy import.** ``librosa`` is imported via :func:`mixing.util.require_package`
  inside the function body, so ``import mixing.audio`` never pulls it. Install it with the
  ``mixing[beats]`` extra.
- **Permissive only.** librosa is ISC (commercial-clean). A future ``backend="madmom"`` /
  ``"beatnet"`` could fill in real downbeats, but madmom's *models* are academic-licensed,
  so librosa stays the default and the sole backend shipped in the extra.
- **Downbeats are best-effort.** librosa has no downbeat tracker, so ``downbeat_times`` is
  empty for ``backend="librosa"``; the field exists so a stronger backend can populate it
  without a signature change. Consumers that want downbeats should fall back to beats when
  the array is empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..util import require_package
from .audio_util import AudioSource, _normalize_audio_source
from .samples import segment_samples

#: Default analysis sample rate. 22.05 kHz is librosa's default and ample for beat/onset.
DEFAULT_SAMPLE_RATE = 22050
#: Default STFT hop for the onset envelope (librosa default). onset_hop_s = hop/​sr.
DEFAULT_HOP_LENGTH = 512


@dataclass(frozen=True)
class BeatGrid:
    """Rhythmic analysis of one audio signal.

    Attributes:
        beat_times: Beat instants in seconds (ascending).
        downbeat_times: Downbeat instants in seconds (best-effort; empty for the
            ``librosa`` backend, which has no downbeat tracker).
        onset_env: The onset-strength envelope (one value per STFT hop; higher = more
            rhythmic onset energy). Its frame *k* is at time ``k * onset_hop_s``.
        onset_hop_s: Seconds between consecutive ``onset_env`` frames (``hop_length/sr``).
        sample_rate: The analysis sample rate.
        tempo_bpm: The estimated global tempo (beats per minute).
    """

    beat_times: np.ndarray
    downbeat_times: np.ndarray
    onset_env: np.ndarray
    onset_hop_s: float
    sample_rate: int
    tempo_bpm: float

    def to_dict(self) -> dict:
        """JSON-round-trippable view (arrays → lists; small enough to inline)."""
        return {
            "beat_times": [round(float(t), 4) for t in self.beat_times],
            "downbeat_times": [round(float(t), 4) for t in self.downbeat_times],
            "onset_hop_s": self.onset_hop_s,
            "sample_rate": self.sample_rate,
            "tempo_bpm": round(float(self.tempo_bpm), 3),
            # onset_env is dense; callers that need it persist the array separately.
        }


def _load_mono_float(source: AudioSource, sample_rate: int) -> np.ndarray:
    """Load any :data:`AudioSource` as a mono float array in ~[-1, 1] at ``sample_rate``."""
    seg = _normalize_audio_source(source, target_type="AudioSegment")
    seg = seg.set_channels(1).set_frame_rate(sample_rate)
    return segment_samples(seg)[:, 0]


def beat_grid(
    audio: AudioSource,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    start_bpm: float = 120.0,
    backend: str = "librosa",
) -> BeatGrid:
    """Estimate beats, (best-effort) downbeats, and the onset envelope of ``audio``.

    Args:
        audio: The audio to analyze (path, numpy array, or ``AudioSegment``).
        sample_rate: Analysis sample rate (mono).
        hop_length: STFT hop for the onset envelope; ``onset_hop_s = hop_length/sr``.
        start_bpm: Tempo prior for the beat tracker (helps on ambiguous material).
        backend: Only ``"librosa"`` (ISC, commercial-clean) is supported. ``"madmom"`` is
            intentionally NOT shipped — its beat models are academic-licensed.

    Returns:
        A :class:`BeatGrid`. ``downbeat_times`` is empty for the librosa backend.

    Raises:
        ValueError: for an unsupported ``backend``.
    """
    if backend != "librosa":
        raise ValueError(
            f"unsupported beat backend {backend!r}; only 'librosa' is shipped "
            "(madmom's beat models are academic-licensed and deliberately excluded)"
        )
    librosa = require_package("librosa")
    y = _load_mono_float(audio, sample_rate)
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        start_bpm=start_bpm,
        units="frames",
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sample_rate, hop_length=hop_length
    )
    tempo_bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return BeatGrid(
        beat_times=np.asarray(beat_times, dtype=np.float64),
        downbeat_times=np.asarray([], dtype=np.float64),  # librosa: no downbeats
        onset_env=np.asarray(onset_env, dtype=np.float32),
        onset_hop_s=hop_length / float(sample_rate),
        sample_rate=sample_rate,
        tempo_bpm=tempo_bpm,
    )
