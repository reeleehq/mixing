"""Structural sections of a song — verse / chorus / bridge by what repeats, not by loudness.

``structural_sections`` answers "where does this song change part?" with a list of
:class:`SongSection` (``start``, ``end``, ``label``). Labels are letters (``"A"``, ``"B"``...):
sections that sound alike share one, so a consumer can tell "the part that comes back"
(a chorus) from "a part that comes once" without this module guessing names.

The default backend (``"librosa"``, ISC) segments a beat/bar-synchronous chroma + MFCC
feature matrix by agglomerative clustering, then clusters the segments by their mean
feature. Boundaries snap to the nearest downbeat when ``downbeats`` are given, so a cut
planner never lands a section change mid-bar. More backends (SongFormer, CC BY 4.0;
MSAF, MIT) register through :func:`register_section_backend`; ``madmom`` and Essentia
models are not shipped (licences).

Heavy imports (librosa, scipy) are lazy, so ``import mixing.audio`` stays light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..util import require_package
from .beats import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE, _load_mono_float
from .audio_util import AudioSource

#: Target seconds per section when ``n_sections`` is not given (a song of 4 min -> ~9).
SECONDS_PER_SECTION = 25.0
MIN_SECTIONS, MAX_SECTIONS = 3, 12
#: Segments whose mean features are closer than this (relative to the spread) share a label.
LABEL_DISTANCE = 0.6


@dataclass(frozen=True)
class SongSection:
    start: float
    end: float
    label: str

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "label": self.label}


SectionBackend = Callable[..., "list[SongSection]"]
SECTION_BACKENDS: dict[str, SectionBackend] = {}


def register_section_backend(
    name: str, fn: SectionBackend, *, replace: bool = False
) -> None:
    """Add a section backend: ``fn(audio, *, downbeats, n_sections, sample_rate) -> [SongSection]``."""
    if name in SECTION_BACKENDS and not replace:
        raise ValueError(f"section backend {name!r} is already registered")
    SECTION_BACKENDS[name] = fn


def _letters(n: int) -> str:
    return chr(ord("A") + n) if n < 26 else f"S{n}"


def snap_to_downbeats(
    times: Sequence[float], downbeats: Sequence[float], *, duration: float
) -> list[float]:
    """Move each boundary to its nearest downbeat; drop duplicates and ends. Pure numpy."""
    down = np.asarray(downbeats, dtype=float)
    out: list[float] = []
    for t in times:
        s = float(down[np.abs(down - t).argmin()]) if len(down) else float(t)
        if 0 < s < duration and (not out or s > out[-1] + 1e-9):
            out.append(s)
    return out


def _librosa_sections(
    audio, *, downbeats, n_sections, sample_rate
) -> "list[SongSection]":
    librosa = require_package("librosa")
    from scipy.cluster.hierarchy import fcluster, linkage

    y = _load_mono_float(audio, sample_rate)
    duration = len(y) / float(sample_rate)
    k = n_sections or int(
        np.clip(round(duration / SECONDS_PER_SECTION), MIN_SECTIONS, MAX_SECTIONS)
    )
    hop = DEFAULT_HOP_LENGTH * 4  # ~93 ms frames: structure moves slowly
    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, hop_length=hop, n_mfcc=13)
    feats = np.vstack(
        [
            librosa.util.normalize(chroma, axis=1),
            librosa.util.normalize(mfcc[1:], axis=1),
        ]
    )
    # smooth over ~2 s so a drum fill is not a boundary
    width = max(1, int(2.0 * sample_rate / hop))
    kernel = np.ones(width) / width
    feats = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, feats)
    k = int(min(k, feats.shape[1]))
    frames = librosa.segment.agglomerative(feats, k)  # start frames of k segments
    bounds = [float(librosa.frames_to_time(f, sr=sample_rate, hop_length=hop)) for f in frames]
    inner = [b for b in bounds if b > 0]
    if len(downbeats):
        inner = snap_to_downbeats(inner, downbeats, duration=duration)
    edges = [0.0, *inner, duration]
    means = []
    for a, b in zip(edges, edges[1:]):
        i0 = int(a * sample_rate / hop)
        i1 = max(i0 + 1, int(b * sample_rate / hop))
        means.append(feats[:, i0:i1].mean(axis=1))
    if len(means) > 1:
        Z = linkage(np.asarray(means), method="average")
        spread = float(Z[:, 2].max()) or 1.0
        ids = fcluster(Z, t=LABEL_DISTANCE * spread, criterion="distance")
    else:
        ids = np.array([1])
    order: dict[int, int] = {}  # label letters in order of first appearance
    for i in ids:
        order.setdefault(int(i), len(order))
    return [
        SongSection(a, b, _letters(order[int(i)]))
        for (a, b), i in zip(zip(edges, edges[1:]), ids)
    ]


register_section_backend("librosa", _librosa_sections)


def structural_sections(
    audio: AudioSource,
    *,
    downbeats: Optional[Sequence[float]] = None,
    n_sections: Optional[int] = None,
    backend: str = "librosa",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[SongSection]:
    """The song's sections in time order, tiling ``[0, duration]``, boundaries on downbeats.

    ``downbeats`` (seconds, e.g. ``beat_grid(..., backend="beat_this").downbeat_times``)
    makes every boundary a downbeat; without them boundaries fall where the features
    changed. ``n_sections`` overrides the song-length default. Raises ``ValueError`` for an
    unregistered ``backend``.
    """
    if backend not in SECTION_BACKENDS:
        raise ValueError(
            f"unsupported section backend {backend!r}; registered: {sorted(SECTION_BACKENDS)}"
        )
    return SECTION_BACKENDS[backend](
        audio,
        downbeats=np.asarray(downbeats if downbeats is not None else [], dtype=float),
        n_sections=n_sections,
        sample_rate=sample_rate,
    )
