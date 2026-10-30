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
- **Permissive only.** librosa is ISC (commercial-clean) and the default. ``backend=
  "beat_this"`` (Beat This!, MIT code and weights) fills in real downbeats when its
  package is installed; madmom's *models* are academic-licensed, so it is not shipped.
  More backends register through :func:`register_beat_backend`.
- **Downbeats are backend-dependent.** librosa has no downbeat tracker, so ``downbeat_times``
  is empty for ``backend="librosa"``.
- **Ranked cuts.** :func:`rank_beats` / :func:`ranked_cut_points` turn the onset envelope
  into "which beats are worth cutting on". Consumers that want downbeats should fall back to beats when
  the array is empty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..util import require_package
from .audio_util import AudioSource, _normalize_audio_source
from .samples import segment_samples

#: Default analysis sample rate. 22.05 kHz is librosa's default and ample for beat/onset.
DEFAULT_SAMPLE_RATE = 22050
#: Default STFT hop for the onset envelope (librosa default). onset_hop_s = hop/​sr.
DEFAULT_HOP_LENGTH = 512
#: Rank bonus a downbeat gets in :func:`rank_beats`.
DOWNBEAT_BONUS = 0.15


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


#: A beat backend: ``(y, sample_rate, hop_length, start_bpm, audio) -> (beats, downbeats,
#: tempo_bpm)``; the onset envelope is computed once, here, for every backend.
BeatBackend = Callable[..., "tuple[np.ndarray, np.ndarray, float]"]
BEAT_BACKENDS: dict[str, BeatBackend] = {}


def register_beat_backend(name: str, fn: BeatBackend, *, replace: bool = False) -> None:
    """Add a beat backend under ``name`` (open-closed seam for ``beat_grid(backend=...)``)."""
    if name in BEAT_BACKENDS and not replace:
        raise ValueError(f"beat backend {name!r} is already registered")
    BEAT_BACKENDS[name] = fn


def _librosa_backend(y, sr, hop_length, start_bpm, audio, *, onset_env):
    librosa = require_package("librosa")
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        units="frames",
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    tempo_bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return (
        np.asarray(beats, dtype=np.float64),
        np.asarray([], dtype=np.float64),  # librosa: no downbeats
        tempo_bpm,
    )


def _beat_this_backend(y, sr, hop_length, start_bpm, audio, *, onset_env):
    """Beat This! (CPJKU, MIT code and weights): beats AND downbeats from a neural net.

    Needs ``beat_this`` (``pip install https://github.com/CPJKU/beat_this/archive/main.zip``,
    plus torch) and, on first use, downloads its checkpoint. It reads the file itself, so
    ``audio`` must be a path.
    """
    try:
        from beat_this.inference import File2Beats
    except ImportError as e:
        raise ImportError(
            "backend='beat_this' needs the `beat_this` package (MIT): "
            "pip install https://github.com/CPJKU/beat_this/archive/main.zip"
        ) from e
    if not isinstance(audio, (str, os.PathLike)):
        raise TypeError("backend='beat_this' needs an audio file path")
    beats, downbeats = File2Beats(device="cpu")(str(audio))
    beats = np.asarray(beats, dtype=np.float64)
    gaps = np.diff(beats)
    tempo = 60.0 / float(np.median(gaps)) if len(gaps) else 0.0
    return beats, np.asarray(downbeats, dtype=np.float64), tempo


register_beat_backend("librosa", _librosa_backend)
register_beat_backend("beat_this", _beat_this_backend)


def beat_grid(
    audio: AudioSource,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    start_bpm: float = 120.0,
    backend: str = "librosa",
    percussive: bool = False,
) -> BeatGrid:
    """Estimate beats, (backend-dependent) downbeats, and the onset envelope of ``audio``.

    Args:
        audio: The audio to analyze (path, numpy array, or ``AudioSegment``).
        sample_rate: Analysis sample rate (mono).
        hop_length: STFT hop for the onset envelope; ``onset_hop_s = hop_length/sr``.
        start_bpm: Tempo prior for the beat tracker (helps on ambiguous material).
        backend: A key of :data:`BEAT_BACKENDS`. ``"librosa"`` (ISC; the default) has no
            downbeat tracker, so ``downbeat_times`` is empty. ``"beat_this"`` (MIT, needs
            the ``beat_this`` package and a file path) returns beats and downbeats.
            ``"madmom"`` is intentionally NOT shipped: its models are academic-licensed.
        percussive: Compute the onset envelope on the percussive (HPSS) component, so
            :func:`rank_beats` ranks drum hits rather than sustained harmonic change.

    Raises:
        ValueError: for an unregistered ``backend``.
    """
    if backend not in BEAT_BACKENDS:
        raise ValueError(
            f"unsupported beat backend {backend!r}; registered: {sorted(BEAT_BACKENDS)} "
            "(madmom's beat models are academic-licensed and deliberately excluded)"
        )
    librosa = require_package("librosa")
    y = _load_mono_float(audio, sample_rate)
    y_onset = librosa.effects.percussive(y) if percussive else y
    onset_env = librosa.onset.onset_strength(
        y=y_onset, sr=sample_rate, hop_length=hop_length
    )
    beat_times, downbeat_times, tempo_bpm = BEAT_BACKENDS[backend](
        y, sample_rate, hop_length, start_bpm, audio, onset_env=onset_env
    )
    return BeatGrid(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        onset_env=np.asarray(onset_env, dtype=np.float32),
        onset_hop_s=hop_length / float(sample_rate),
        sample_rate=sample_rate,
        tempo_bpm=tempo_bpm,
    )


def rank_beats(grid: BeatGrid) -> np.ndarray:
    """Each beat's cut-worthiness in ``[0, 1]``: its onset strength, rank-normalised.

    The onset envelope's peak within half a hop-neighbourhood of the beat, turned into a
    percentile among the song's beats (1.0 = the strongest), so a quiet song and a loud
    one rank alike. Downbeats get a small bonus (editors cut on the one). Pure numpy.
    """
    beats = np.asarray(grid.beat_times, dtype=float)
    if not len(beats):
        return np.zeros(0)
    env = np.asarray(grid.onset_env, dtype=float)
    if not len(env):
        return np.full(len(beats), 0.5)
    centre = np.clip(np.round(beats / grid.onset_hop_s).astype(int), 0, len(env) - 1)
    peak = np.array([env[max(0, c - 1) : c + 2].max() for c in centre])
    rank = peak.argsort().argsort() / max(len(beats) - 1, 1)  # percentile, 0..1
    down = np.asarray(grid.downbeat_times, dtype=float)
    if len(down):
        near = np.abs(beats[:, None] - down[None, :]).min(axis=1) < grid.onset_hop_s
        rank = np.where(near, np.minimum(1.0, rank + DOWNBEAT_BONUS), rank)
    return rank


def ranked_cut_points(grid: BeatGrid, *, keep: float = 0.5) -> np.ndarray:
    """The ``keep`` fraction (0..1] of beats most worth cutting on, in time order.

    Professional editors skip most beats; this is the subset a planner can cut on
    instead of a fixed density. ``keep=1`` returns every beat.
    """
    if not 0 < keep <= 1:
        raise ValueError(f"keep must be in (0, 1], got {keep}")
    beats = np.asarray(grid.beat_times, dtype=float)
    if not len(beats):
        return beats
    n = max(1, int(round(keep * len(beats))))
    top = np.argsort(-rank_beats(grid), kind="stable")[:n]
    return beats[np.sort(top)]
