"""
Split a long audio into segments — for example, the individual songs in a
concert recording, a DJ mix, or a radio show.

Three things make this hard in practice:

1. Sometimes the boundaries are obvious (silence between tracks).
2. Sometimes there's continuous background noise (audience, room tone),
   so the energy never really drops.
3. Sometimes the boundaries are between different *kinds* of audio
   (speech vs music on the radio).

This module offers a small family of pluggable strategies that cover those
three regimes, plus a single entry point ``find_segments`` that you can
parametrize for your particular case. It is not magic — it is a toolbox
with sensible defaults. Tuning the keyword arguments matters.

The strategies, in increasing order of sophistication:

- ``"silence"`` — pydub silence detection. Good for clean track separations.
- ``"energy_novelty"`` — adaptive RMS valley detection. Catches fade-outs and
  quiet inter-track moments that aren't quite silence.
- ``"self_similarity"`` — Foote's checkerboard novelty (ICME 2000) on
  log-spectrogram features. The standard tool for concert recordings where
  amplitude is roughly constant: it detects boundaries where the *spectral
  content* changes abruptly. See Foote, "Automatic audio segmentation using
  a measure of audio novelty" (ICME 2000).
- ``"speech_music"`` — low-energy frame ratio + ZCR variance (Scheirer &
  Slaney 1997-style handcrafted features). Cheap and dependency-free, and fine
  for speech vs *instrumental* music, but it does **not** separate talk from
  *sung* songs: sung vocals look just as syllabic as speech. On a real radio
  show (DJ talk, jingles, sung pop songs) a 4 Hz envelope-modulation feature
  measured 0.529±0.159 on speech vs 0.535±0.158 on music.
- ``"audioset"`` — a pretrained AudioSet tagger (by default the Audio
  Spectrogram Transformer) scores speech vs music in sliding windows; runs of
  music become segments, their edges snapped to the quietest nearby moment.
  This is the strategy for **radio shows and tapes**: it tells sung songs from
  talk, and with ``min_song_duration`` it labels long music runs ``"song"``
  and short ones (stingers, station IDs) ``"jingle"``. The model is a seam
  (``classifier=``); its default needs the ``mixing[ml]`` extra (``torch``,
  ``transformers``), imported lazily. See :mod:`mixing.audio.tagging`.

You can also pass your own callable as ``strategy`` for custom logic.

Hosts usually announce a song just before it starts; :func:`text_before_segments`
returns the transcript words just before each segment (pair it with
:func:`mixing.srt.parse_vtt_words` for YouTube captions, or a Scribe transcript).
To find where that talk overlaps the carved song itself (an intro over the opening
bars, a back-announce over the fade), use :func:`mixing.audio.talk_edges.find_talk_edges`.

Output:
- ``find_segments`` returns a list of ``Segment`` dataclasses (start/end in
  seconds), which you can hand to a player, a transcript tool, or whatever.
- ``extract_segments`` does the same plus exports each segment as a file.
- ``Segment.as_offset_duration()`` and ``.as_start_end()`` give the two
  common timestamp representations.

Examples:
    >>> from mixing.audio import find_segments, extract_segments  # doctest: +SKIP
    >>>
    >>> # DJ mix with silences between tracks
    >>> segs = find_segments("mix.mp3", strategy="silence",
    ...                       silence_thresh_db=-40, min_silence_len=1.5)  # doctest: +SKIP
    >>>
    >>> # Concert: continuous audience noise, songs differ in spectral content
    >>> segs = find_segments("concert.wav", strategy="self_similarity",
    ...                       kernel_seconds=12.0,
    ...                       min_peak_distance_seconds=60.0)  # doctest: +SKIP
    >>>
    >>> # Radio show / tape with sung songs: learned speech vs music (mixing[ml])
    >>> segs = find_segments("show.m4a", strategy="audioset",
    ...                       min_song_duration=75)  # doctest: +SKIP
    >>> songs = [s for s in segs if s.label == "song"]  # doctest: +SKIP
    >>>
    >>> # ... and what the host said just before each song
    >>> from mixing.srt import parse_vtt_words  # doctest: +SKIP
    >>> words = parse_vtt_words(open("show.en.vtt").read())  # doctest: +SKIP
    >>> intros = text_before_segments(songs, words)  # doctest: +SKIP
    >>>
    >>> # Save them to disk
    >>> paths = extract_segments("concert.wav", segs, output="songs/")  # doctest: +SKIP
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, Union, TYPE_CHECKING
import numpy as np

from ..util import require_package
from .audio_util import AudioSource, _normalize_audio_source
from .samples import segment_samples
from .tagging import AST_SAMPLE_RATE, AudioClassifier, ast_audioset_classifier

if TYPE_CHECKING:
    from pydub import AudioSegment


SegmentationStrategyName = Literal[
    "silence", "energy_novelty", "self_similarity", "speech_music", "audioset"
]

#: ``"audioset"``: a window is music if P(music) exceeds this (and P(speech)).
DEFAULT_MUSIC_THRESHOLD = 0.5
#: ``"audioset"``: median-filter length, in windows, applied to the music/talk labels.
DEFAULT_SMOOTH_WINDOWS = 5
#: ``"audioset"``: boundaries move to the quietest moment within this many seconds.
DEFAULT_SNAP_RADIUS_SECONDS = 4.0
#: Length (seconds) of the RMS frames scanned when snapping a boundary.
SNAP_FRAME_SECONDS = 0.1
_SNAP_TOLERANCE = 1e-6  # seconds; absorbs float error when two boundaries coincide
#: Typical radio-show split: music runs at least this long are songs, shorter ones jingles.
RADIO_MIN_SONG_SECONDS = 75.0
#: :func:`text_before_segments`: seconds of transcript taken before a segment start.
DEFAULT_TEXT_BEFORE_SECONDS = 25.0
#: :func:`text_before_segments`: seconds of transcript taken after a segment start.
DEFAULT_TEXT_AFTER_SECONDS = 3.0


@dataclass(frozen=True)
class Segment:
    """A time-bounded slice of an audio file.

    Attributes:
        start: Segment start in seconds.
        end: Segment end in seconds.
        label: Optional tag (e.g. ``"speech"``, ``"music"``, ``"song"``).
        score: Optional confidence/novelty value. Higher = stronger boundary.
    """

    start: float
    end: float
    label: str | None = None
    score: float | None = None

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start

    @property
    def offset(self) -> float:
        """Alias for ``start`` — read-only."""
        return self.start

    def as_start_end(self) -> tuple[float, float]:
        """Return ``(start, end)`` in seconds."""
        return (self.start, self.end)

    def as_offset_duration(self) -> tuple[float, float]:
        """Return ``(offset, duration)`` in seconds."""
        return (self.start, self.duration)


# ----------------------------------------------------------------------------
# Strategy: silence-based
# ----------------------------------------------------------------------------


def segment_by_silence(
    audio: "AudioSegment",
    *,
    silence_thresh_db: float = -40.0,
    min_silence_len: float = 1.0,
    seek_step: float = 0.01,
    keep_silence: float = 0.0,
    label: str = "non_silent",
) -> list[Segment]:
    """Find non-silent regions using pydub's silence detector.

    Best for cleanly separated tracks (DJ mixes, audiobooks, voicemails) where
    there's a real silence gap between segments.

    Args:
        audio: pydub ``AudioSegment``.
        silence_thresh_db: dBFS threshold below which audio counts as silence.
            More negative = stricter silence requirement. Tune downward for
            noisy recordings (e.g. -50 dBFS).
        min_silence_len: Minimum silence duration in seconds to count as a
            boundary. Tracks separated by less than this are merged.
        seek_step: Search granularity in seconds.
        keep_silence: Seconds of surrounding silence to include with each
            non-silent segment.
        label: Label assigned to each returned segment.

    Returns:
        List of ``Segment`` covering the non-silent regions.
    """
    silence_mod = require_package("pydub.silence")
    duration_ms = len(audio)
    nonsilent_ranges = silence_mod.detect_nonsilent(
        audio,
        min_silence_len=int(min_silence_len * 1000),
        silence_thresh=silence_thresh_db,
        seek_step=max(1, int(seek_step * 1000)),
    )
    keep_ms = int(keep_silence * 1000)
    segments = []
    for start_ms, end_ms in nonsilent_ranges:
        s = max(0, start_ms - keep_ms)
        e = min(duration_ms, end_ms + keep_ms)
        segments.append(Segment(start=s / 1000.0, end=e / 1000.0, label=label))
    return segments


# ----------------------------------------------------------------------------
# Shared signal helpers
# ----------------------------------------------------------------------------


def _to_mono_floats(
    audio: "AudioSegment", *, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Downmix to mono, resample, return float64 in [-1, 1] and the sample rate."""
    seg = audio.set_channels(1).set_frame_rate(sample_rate)
    return segment_samples(seg)[:, 0], sample_rate


def _frame_rms(
    samples: np.ndarray, *, sr: int, frame_seconds: float, hop_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms_per_frame, frame_center_times_in_seconds)."""
    frame_len = max(1, int(frame_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    if len(samples) < frame_len:
        return np.array(
            [np.sqrt(np.mean(samples**2)) if len(samples) else 0.0]
        ), np.array([0.0])
    n_frames = (len(samples) - frame_len) // hop_len + 1
    # Vectorized framing via stride tricks
    shape = (n_frames, frame_len)
    strides = (samples.strides[0] * hop_len, samples.strides[0])
    framed = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(framed**2, axis=1))
    times = np.arange(n_frames) * hop_seconds + frame_seconds / 2.0
    return rms, times


def _smooth(signal: np.ndarray, window_n: int) -> np.ndarray:
    """Hanning-window smoothing. Returns ``signal`` unchanged if window_n < 3.

    ``np.hanning`` yields all zeros for window sizes < 3, so smoothing with
    such a window would produce NaNs.
    """
    if window_n < 3 or len(signal) == 0:
        return signal
    window = np.hanning(window_n)
    s = window.sum()
    if s <= 0:
        return signal
    window = window / s
    return np.convolve(signal, window, mode="same")


def _boundaries_to_segments(
    boundary_times: Sequence[float],
    *,
    total_duration: float,
    label: str | None = None,
    scores: Sequence[float] | None = None,
) -> list[Segment]:
    """Convert sorted boundary timestamps to back-to-back segments."""
    bounds = [0.0] + sorted(float(t) for t in boundary_times) + [total_duration]
    if scores is None:
        scores_seq: list[float | None] = [None] * (len(bounds) - 1)
    else:
        # Pad scores to match number of segments (first segment has no leading boundary)
        scores_seq = [None] + list(scores) + [None]
        scores_seq = scores_seq[: len(bounds) - 1]
    segs = []
    for i in range(len(bounds) - 1):
        if bounds[i + 1] - bounds[i] <= 0:
            continue
        segs.append(
            Segment(
                start=bounds[i], end=bounds[i + 1], label=label, score=scores_seq[i]
            )
        )
    return segs


# ----------------------------------------------------------------------------
# Strategy: energy novelty (RMS valley detection)
# ----------------------------------------------------------------------------


def segment_by_energy(
    audio: "AudioSegment",
    *,
    sample_rate: int = 16000,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.1,
    smooth_seconds: float = 2.0,
    valley_threshold_factor: float = 0.5,
    min_peak_distance_seconds: float = 30.0,
    label: str = "segment",
) -> list[Segment]:
    """Split audio at local energy valleys (fade-outs and quiet moments).

    Computes frame-wise RMS, smooths it, and finds local minima that fall
    below a fraction of the local maximum. Useful when tracks fade into each
    other without true silence.

    Args:
        audio: pydub ``AudioSegment``.
        sample_rate: Analysis sample rate (Hz). Lower is faster.
        frame_seconds: RMS frame length in seconds.
        hop_seconds: RMS hop length in seconds.
        smooth_seconds: Smoothing window length in seconds. Bigger smooths
            out within-song dynamics.
        valley_threshold_factor: A frame is a valley candidate only if its
            RMS is below ``factor * max(RMS)``. ``0.5`` = halfway down from
            the peak. Lower this for cleaner tracks, raise it for messier ones.
        min_peak_distance_seconds: Minimum spacing between valleys. Prevents
            multiple boundaries inside one inter-track pause and acts as a
            crude minimum-song-length filter.
        label: Label assigned to every returned segment.

    Returns:
        List of back-to-back ``Segment``s covering the whole audio.
    """
    find_peaks = require_package("scipy.signal").find_peaks

    samples, sr = _to_mono_floats(audio, sample_rate=sample_rate)
    rms, times = _frame_rms(
        samples, sr=sr, frame_seconds=frame_seconds, hop_seconds=hop_seconds
    )
    if len(rms) <= 2:
        return [Segment(0.0, len(samples) / sr, label=label)]

    smooth_n = max(1, int(smooth_seconds / hop_seconds))
    rms = _smooth(rms, smooth_n)

    # Find valleys = peaks of -RMS, capped by threshold factor
    rms_max = float(np.max(rms))
    if rms_max <= 0:
        return [Segment(0.0, len(samples) / sr, label=label)]
    inverted = -rms
    height_threshold = -valley_threshold_factor * rms_max
    distance = max(1, int(min_peak_distance_seconds / hop_seconds))
    peaks, props = find_peaks(inverted, height=height_threshold, distance=distance)

    boundary_times = times[peaks].tolist()
    # Score = how deep the valley is, normalized
    scores = ((rms_max - rms[peaks]) / rms_max).tolist() if len(peaks) else []
    return _boundaries_to_segments(
        boundary_times,
        total_duration=len(samples) / sr,
        label=label,
        scores=scores,
    )


# ----------------------------------------------------------------------------
# Strategy: self-similarity matrix novelty (Foote's checkerboard)
# ----------------------------------------------------------------------------


def _foote_checkerboard_kernel(half_size: int) -> np.ndarray:
    """Foote's Gaussian-tapered checkerboard kernel of size (2*L)x(2*L).

    Quadrants: top-left and bottom-right are positive (intra-segment
    coherence), top-right and bottom-left are negative (cross-segment
    contrast). Tapered with a 2-D Gaussian (std = L/3) so frames near the
    boundary contribute most.
    """
    L = max(1, int(half_size))
    M = 2 * L
    # Sign vector: -1 for first half, +1 for second half
    sign = np.concatenate([-np.ones(L), np.ones(L)])
    sign_kernel = np.outer(sign, sign)
    # Gaussian taper
    grid = np.arange(M) - (M - 1) / 2.0
    sigma = max(L / 3.0, 1.0)
    g = np.exp(-(grid**2) / (2 * sigma**2))
    taper = np.outer(g, g)
    kernel = sign_kernel * taper
    return kernel / np.abs(kernel).sum()


def _log_spectrogram_features(
    samples: np.ndarray,
    *,
    sr: int,
    n_fft: int,
    hop_seconds: float,
    n_bands: int = 40,
) -> tuple[np.ndarray, float]:
    """Compute a simple log-magnitude spectrogram with band-averaging.

    Returns features of shape ``(n_bands, n_frames)`` and the actual hop
    in seconds.
    """
    stft = require_package("scipy.signal").stft
    hop_len = max(1, int(hop_seconds * sr))
    noverlap = max(0, n_fft - hop_len)
    f, t, Z = stft(
        samples,
        fs=sr,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    log_mag = np.log1p(np.abs(Z))  # (n_freq, n_frames)

    # Average into n_bands log-spaced frequency bands. Avoids needing a real
    # mel filterbank while still giving timbre-sensitive features.
    n_freq = log_mag.shape[0]
    if n_bands >= n_freq:
        feats = log_mag
    else:
        # Log-spaced bin edges, skipping DC.
        edges = np.unique(np.geomspace(2, n_freq, num=n_bands + 1).astype(int))
        if len(edges) < n_bands + 1:
            # Fallback to linear if log-spacing collapses (very small n_freq)
            edges = np.linspace(1, n_freq, num=n_bands + 1).astype(int)
        feats = np.stack(
            [
                log_mag[edges[i] : edges[i + 1]].mean(axis=0)
                for i in range(len(edges) - 1)
            ],
            axis=0,
        )

    # Z-score each band across time so loud bands don't dominate the SSM.
    mean = feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True) + 1e-9
    feats = (feats - mean) / std
    actual_hop = (t[1] - t[0]) if len(t) > 1 else hop_seconds
    return feats, float(actual_hop)


def segment_by_self_similarity(
    audio: "AudioSegment",
    *,
    sample_rate: int = 11025,
    n_fft: int = 2048,
    hop_seconds: float = 0.25,
    n_bands: int = 40,
    kernel_seconds: float = 12.0,
    novelty_smooth_seconds: float = 4.0,
    peak_threshold_factor: float = 1.0,
    min_peak_distance_seconds: float = 30.0,
    label: str = "song",
) -> list[Segment]:
    """Find boundaries via Foote's checkerboard novelty on the SSM.

    This is the standard approach for finding song boundaries in a recording
    where amplitude is roughly constant (concerts, live sets, continuous
    radio with steady levels). It looks at *what* the audio sounds like
    rather than *how loud* it is, so it works when energy-based methods fail.

    Pipeline:
      1. log-magnitude spectrogram, averaged into log-spaced bands, z-scored
      2. cosine self-similarity matrix
      3. correlate the SSM diagonal with a Gaussian-tapered checkerboard
         kernel — this measures how much frame ``t`` is "the end of one
         block and the start of another"
      4. peaks in the resulting novelty curve are segment boundaries

    Args:
        audio: pydub ``AudioSegment``.
        sample_rate: Analysis sample rate (Hz). 11025 is plenty for novelty.
        n_fft: STFT window length in samples.
        hop_seconds: STFT hop length in seconds.
        n_bands: Number of log-spaced frequency bands. 20-60 is reasonable.
        kernel_seconds: Half-width of the checkerboard kernel times 2, i.e.
            roughly how much context (in seconds) is used on each side of a
            candidate boundary. Section-level boundaries (songs) usually
            want 8-20 s.
        novelty_smooth_seconds: Smoothing applied to the novelty curve
            before peak picking. Reduces jitter from beat-level changes.
        peak_threshold_factor: Adaptive threshold = mean + factor * std of
            the novelty curve. Raise to be stricter, lower to find more
            boundaries.
        min_peak_distance_seconds: Minimum spacing between detected
            boundaries. Acts as a minimum-song-length filter.
        label: Label assigned to every returned segment.

    Returns:
        List of back-to-back ``Segment``s covering the whole audio. Each
        segment's ``score`` is the novelty value at its trailing boundary.
    """
    find_peaks = require_package("scipy.signal").find_peaks

    samples, sr = _to_mono_floats(audio, sample_rate=sample_rate)
    if len(samples) == 0:
        return []
    feats, actual_hop = _log_spectrogram_features(
        samples, sr=sr, n_fft=n_fft, hop_seconds=hop_seconds, n_bands=n_bands
    )
    n_frames = feats.shape[1]
    if n_frames < 4:
        return [Segment(0.0, len(samples) / sr, label=label)]

    # Cosine similarity = dot-product of L2-normalized columns
    norms = np.linalg.norm(feats, axis=0, keepdims=True) + 1e-12
    feats_n = feats / norms
    S = feats_n.T @ feats_n  # (n_frames, n_frames)

    # Foote kernel
    L = max(2, int(round(kernel_seconds / 2.0 / actual_hop)))
    kernel = _foote_checkerboard_kernel(L)
    M = kernel.shape[0]  # = 2L

    # Slide kernel along the diagonal. Cheap loop: novelty length is small.
    novelty = np.zeros(n_frames, dtype=np.float64)
    for i in range(L, n_frames - L):
        novelty[i] = np.sum(kernel * S[i - L : i + L, i - L : i + L])

    # Smooth and pick peaks
    smooth_n = max(1, int(novelty_smooth_seconds / actual_hop))
    novelty = _smooth(novelty, smooth_n)
    nov_mean = float(novelty.mean())
    nov_std = float(novelty.std())
    height = nov_mean + peak_threshold_factor * nov_std
    distance = max(1, int(min_peak_distance_seconds / actual_hop))
    peak_idx, _ = find_peaks(novelty, height=height, distance=distance)

    boundary_times = (peak_idx * actual_hop).tolist()
    scores = novelty[peak_idx].tolist()
    return _boundaries_to_segments(
        boundary_times,
        total_duration=len(samples) / sr,
        label=label,
        scores=scores,
    )


# ----------------------------------------------------------------------------
# Strategy: speech vs music classification
# ----------------------------------------------------------------------------


def _zero_crossing_rate(framed: np.ndarray) -> np.ndarray:
    """ZCR per frame, shape (n_frames,)."""
    signs = np.sign(framed)
    signs[signs == 0] = 1.0
    return np.mean(np.abs(np.diff(signs, axis=1)), axis=1) / 2.0


def _frame_signal(samples: np.ndarray, *, frame_len: int, hop_len: int) -> np.ndarray:
    """Frame a 1-D signal into a 2-D array of overlapping frames."""
    if len(samples) < frame_len:
        return samples[np.newaxis, :].astype(np.float64)
    n_frames = (len(samples) - frame_len) // hop_len + 1
    shape = (n_frames, frame_len)
    strides = (samples.strides[0] * hop_len, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)


def segment_by_speech_music(
    audio: "AudioSegment",
    *,
    sample_rate: int = 16000,
    frame_seconds: float = 0.05,
    hop_seconds: float = 0.025,
    window_seconds: float = 1.5,
    low_energy_threshold: float = 0.5,
    speech_decision_threshold: float = 0.5,
    smooth_seconds: float = 5.0,
    min_segment_duration: float = 3.0,
    speech_label: str = "speech",
    music_label: str = "music",
) -> list[Segment]:
    """Tag regions of audio as ``"speech"`` or ``"music"``.

    Useful for podcasts with *instrumental* interludes. It does not separate
    talk from sung songs (sung vocals look like speech to these features);
    for radio shows use ``segment_by_audioset`` (``strategy="audioset"``). Implements
    a simplified version of Scheirer & Slaney (ICASSP 1997) using two of
    their most discriminative features:

    - **Low-energy frame ratio (LEFR)**: fraction of short frames in a
      ~1-2 s window with RMS below half the window mean. Speech has
      bursts of silence between syllables, so its LEFR is high; music
      stays loud, so its LEFR is low.
    - **ZCR variance**: speech alternates voiced/unvoiced (low/high ZCR);
      music's ZCR is steadier.

    These are combined into a soft speech score, smoothed, and thresholded.
    Consecutive frames with the same class become a single segment.

    Args:
        audio: pydub ``AudioSegment``.
        sample_rate: Analysis sample rate (Hz).
        frame_seconds: Short-frame length for RMS/ZCR (e.g. 50 ms).
        hop_seconds: Short-frame hop length.
        window_seconds: Long-window length over which LEFR and ZCR-variance
            are aggregated. ~1-2 s is the canonical choice.
        low_energy_threshold: A short frame counts as "low energy" if its
            RMS is below ``low_energy_threshold * mean RMS in window``.
            Scheirer & Slaney used 0.5; lower values (~0.1) only count true
            silence frames, raise it for noisier recordings.
        speech_decision_threshold: Speech score threshold in [0, 1] above
            which a window is tagged as speech. Raise to make speech-tagging
            stricter.
        smooth_seconds: Smoothing applied to the soft speech score before
            thresholding. Larger = fewer, longer segments.
        min_segment_duration: Drop or merge segments shorter than this.
        speech_label, music_label: Labels assigned to each region.

    Returns:
        List of back-to-back ``Segment``s tagging speech and music regions.
    """
    samples, sr = _to_mono_floats(audio, sample_rate=sample_rate)
    if len(samples) == 0:
        return []
    total_duration = len(samples) / sr

    frame_len = max(1, int(frame_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    framed = _frame_signal(samples, frame_len=frame_len, hop_len=hop_len)
    short_rms = np.sqrt(np.mean(framed**2, axis=1) + 1e-20)
    short_zcr = _zero_crossing_rate(framed)
    n_short = len(short_rms)
    if n_short < 4:
        return [Segment(0.0, total_duration, label=music_label)]

    # Aggregate over long windows: LEFR + ZCR variance.
    win_n = max(2, int(window_seconds / hop_seconds))
    step_n = max(1, win_n // 2)
    n_long = max(1, (n_short - win_n) // step_n + 1)

    speech_score = np.zeros(n_long, dtype=np.float64)
    long_times = np.zeros(n_long, dtype=np.float64)
    for i in range(n_long):
        a = i * step_n
        b = min(n_short, a + win_n)
        rms_win = short_rms[a:b]
        zcr_win = short_zcr[a:b]
        mean_rms = rms_win.mean() + 1e-12
        lefr = float(np.mean(rms_win < low_energy_threshold * mean_rms))
        # ZCR variance, scaled into ~[0, 1] empirically (speech ~0.01-0.05).
        zcr_var = float(zcr_win.var())
        zcr_var_norm = min(1.0, zcr_var / 0.005)
        # Soft speech score: average of the two cues.
        speech_score[i] = 0.6 * lefr + 0.4 * zcr_var_norm
        long_times[i] = (a + b) / 2.0 * hop_seconds

    # Smooth and threshold.
    smooth_steps = max(1, int(smooth_seconds / (step_n * hop_seconds)))
    speech_score = _smooth(speech_score, smooth_steps)
    is_speech = speech_score > speech_decision_threshold

    # Walk the labels and emit segments at every class change.
    segments: list[Segment] = []
    cur_class = bool(is_speech[0])
    cur_start = 0.0
    for i in range(1, n_long):
        if bool(is_speech[i]) != cur_class:
            boundary = float(long_times[i])
            segments.append(
                Segment(
                    start=cur_start,
                    end=boundary,
                    label=speech_label if cur_class else music_label,
                    score=float(speech_score[i - 1]),
                )
            )
            cur_start = boundary
            cur_class = bool(is_speech[i])
    segments.append(
        Segment(
            start=cur_start,
            end=total_duration,
            label=speech_label if cur_class else music_label,
            score=float(speech_score[-1]),
        )
    )

    if min_segment_duration > 0:
        segments = _absorb_short_segments(segments, min_segment_duration)
    return segments


# ----------------------------------------------------------------------------
# Strategy: learned speech vs music (AudioSet tagger)
# ----------------------------------------------------------------------------


def _label_runs(flags: np.ndarray) -> list[tuple[bool, int, int]]:
    """Runs of equal values as ``(value, first_index, last_index_exclusive)``.

    >>> _label_runs(np.array([0, 0, 1, 1, 1, 0]))
    [(False, 0, 2), (True, 2, 5), (False, 5, 6)]
    """
    runs, start = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            runs.append((bool(flags[start]), start, i))
            start = i
    return runs


def _mono_float32(
    audio: "AudioSegment", *, sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1]; resampled (anti-aliased) only if asked."""
    seg = audio.set_channels(1)
    x = segment_samples(seg, dtype=np.float32)[:, 0]
    sr = seg.frame_rate
    if sample_rate is not None and sample_rate != sr and len(x):
        g = np.gcd(int(sr), int(sample_rate))
        resample_poly = require_package("scipy.signal").resample_poly
        x = resample_poly(x, sample_rate // g, sr // g).astype(np.float32)
        sr = sample_rate
    return x, sr


def _snap_boundaries(
    boundaries: Sequence[float],
    samples: np.ndarray,
    *,
    sr: int,
    radius: float,
    total_duration: float,
) -> list[float]:
    """Move each boundary to the quietest ``SNAP_FRAME_SECONDS`` frame within ``radius``.

    Boundaries stay ordered: each one searches only between the (already
    snapped) previous boundary and the next raw one. Two may coincide; the
    caller drops the empty segment between them.
    """
    if radius <= 0 or not boundaries:
        return list(boundaries)
    snapped: list[float] = []
    for k, b in enumerate(boundaries):
        lo = max(b - radius, snapped[-1] if snapped else 0.0)
        hi = min(
            b + radius, boundaries[k + 1] if k + 1 < len(boundaries) else total_duration
        )
        # Start half a frame early so a frame is centred on ``lo`` itself.
        a = max(0, int(round((lo - SNAP_FRAME_SECONDS / 2) * sr)))
        rms, centers = _frame_rms(
            samples[a : int(hi * sr)],
            sr=sr,
            frame_seconds=SNAP_FRAME_SECONDS,
            hop_seconds=SNAP_FRAME_SECONDS / 2,
        )
        centers = centers + a / sr
        # ``>=``: two boundaries may land on the same quiet point, which drops
        # the sliver between them (e.g. a few words between a jingle and a song).
        mask = (centers >= lo - _SNAP_TOLERANCE) & (centers < hi)
        best = float(centers[mask][np.argmin(rms[mask])]) if mask.any() else b
        snapped.append(max(best, lo))
    return snapped


def segment_by_audioset(
    audio: "AudioSegment",
    *,
    classifier: AudioClassifier | None = None,
    sample_rate: int | None = AST_SAMPLE_RATE,
    music_threshold: float = DEFAULT_MUSIC_THRESHOLD,
    smooth_windows: int = DEFAULT_SMOOTH_WINDOWS,
    snap_radius: float = DEFAULT_SNAP_RADIUS_SECONDS,
    min_song_duration: float | None = None,
    speech_label: str = "speech",
    music_label: str = "music",
    song_label: str = "song",
    jingle_label: str = "jingle",
) -> list[Segment]:
    """Split audio into talk and music with a pretrained audio tagger.

    The strategy for radio shows and tapes where songs are *sung*: handcrafted
    features (``"speech_music"``) confuse sung vocals with speech, a model
    trained on AudioSet does not.

    Pipeline:
      1. ``classifier(samples, sr)`` scores P(speech) and P(music) per window
         (default: the AudioSet AST, 5 s windows every 2 s);
      2. a window is music if ``P(music) > music_threshold`` and
         ``P(music) > P(speech)``; the labels are median-filtered over
         ``smooth_windows`` windows;
      3. runs of equal labels become back-to-back segments. A music run
         extends to the outer edges of its outermost music windows, so songs
         are not clipped;
      4. each boundary moves to the quietest moment within ``snap_radius``
         seconds (the gap between a song and the host's voice);
      5. with ``min_song_duration``, music segments at least that long are
         labelled ``song_label`` and shorter ones ``jingle_label``.

    Args:
        audio: pydub ``AudioSegment``.
        classifier: The seam: any ``(samples, sr) -> WindowProbabilities``
            (see :mod:`mixing.audio.tagging`). ``None`` uses
            :func:`~mixing.audio.tagging.ast_audioset_classifier`, which needs
            ``pip install 'mixing[ml]'``.
        sample_rate: Rate (Hz) the audio is resampled to, with an anti-aliased
            polyphase filter, before classifying and snapping. The default is
            the AST's 16 kHz (it also keeps memory low on long tapes); ``None``
            keeps the native rate and leaves resampling to the classifier. Not
            pydub's resampler: it is not anti-aliased, and on a real tape it
            lowered the tagger's music probabilities from ~0.8 to ~0.5.
        music_threshold: Minimum P(music) for a window to count as music.
        smooth_windows: Median-filter length in windows (``1`` disables it;
            even values are rounded up to odd). At the default 2 s hop, music
            runs of about 7 s or less are smoothed away, so very short stingers
            never become jingles. It counts windows, so retune it (and
            ``snap_radius``) for a classifier with a different window/hop.
        snap_radius: Seconds a boundary may move to reach a quiet point
            (``0`` disables snapping). It picks the quietest 100 ms in reach,
            which is usually the gap between a song and the host but can be a
            pause inside a song; lower it if songs come out trimmed.
        min_song_duration: Seconds. If given, split music segments into
            songs (>=) and jingles (<); ``RADIO_MIN_SONG_SECONDS`` (75) suits
            radio shows. ``None`` labels them all ``music_label``.
        speech_label: Label of non-music segments (talk, but also silence or
            noise: anything the tagger does not call music).
        music_label, song_label, jingle_label: Labels of music segments.

    Returns:
        Back-to-back ``Segment``s covering the whole audio. ``score`` is the
        mean probability of the segment's class over its windows. Two speech
        segments are never adjacent; two music segments can be (a jingle
        running straight into a song).

    >>> from mixing.audio.tagging import WindowProbabilities
    >>> def stub(samples, sr):  # music from 10 s to 30 s, talk elsewhere
    ...     t = np.arange(0.0, 36.0, 2.0)
    ...     m = ((t >= 10) & (t + 4 <= 30)).astype(float)
    ...     return WindowProbabilities(t, t + 4, 1 - m, m)
    >>> from pydub import AudioSegment
    >>> segs = segment_by_audioset(
    ...     AudioSegment.silent(40_000), classifier=stub, snap_radius=0,
    ...     min_song_duration=15,
    ... )
    >>> [(s.label, s.start, s.end) for s in segs]
    [('speech', 0.0, 10.0), ('song', 10.0, 30.0), ('speech', 30.0, 40.0)]
    """
    samples, sr = _mono_float32(audio, sample_rate=sample_rate)
    if len(samples) == 0:
        return []
    total_duration = len(samples) / sr
    classify = classifier if classifier is not None else ast_audioset_classifier()
    probs = classify(samples, sr)
    if len(probs) == 0:
        return [Segment(0.0, total_duration, label=speech_label)]

    speech, music = np.asarray(probs.speech), np.asarray(probs.music)
    is_music = (music > music_threshold) & (music > speech)
    if smooth_windows > 1 and len(is_music) > 1:
        median_filter = require_package("scipy.ndimage").median_filter
        size = smooth_windows | 1  # an even median window would bias and shift
        is_music = median_filter(is_music.astype(np.int8), size=size, mode="nearest")
    runs = _label_runs(np.asarray(is_music, dtype=bool))

    # A boundary between two runs sits at the music run's outer window edge.
    raw = [
        float(probs.ends[left_end - 1])
        if left_music
        else float(probs.starts[right_start])
        for (left_music, _, left_end), (_, right_start, _) in zip(runs, runs[1:])
    ]
    raw = [min(max(b, 0.0), total_duration) for b in raw]
    # Drop runs squeezed to nothing by their neighbours' window overlap.
    kept_runs, kept_bounds = [runs[0]], []
    for run, b in zip(runs[1:], raw):
        while kept_bounds and b <= kept_bounds[-1]:
            kept_bounds.pop()
            kept_runs.pop()
        if kept_runs[-1][0] == run[0]:
            continue  # merged into the previous run of the same kind
        kept_runs.append(run)
        kept_bounds.append(b)
    bounds = _snap_boundaries(
        kept_bounds, samples, sr=sr, radius=snap_radius, total_duration=total_duration
    )
    edges = [0.0] + bounds + [total_duration]

    # Snapping can collapse a run to nothing; drop it, and re-join the talk on
    # either side. Two music runs left adjacent stay two items (e.g. a jingle
    # straight into a song).
    spans: list[list] = []
    for (is_mus, _, _), start, end in zip(kept_runs, edges, edges[1:]):
        if end <= start:
            continue
        if spans and not is_mus and not spans[-1][0]:
            spans[-1][2] = end
        else:
            spans.append([is_mus, start, end])

    centers = (np.asarray(probs.starts) + np.asarray(probs.ends)) / 2
    segments = []
    for is_mus, start, end in spans:
        inside = (centers >= start) & (centers <= end)
        cls = music if is_mus else speech
        score = float(cls[inside].mean()) if inside.any() else None
        if not is_mus:
            label = speech_label
        elif min_song_duration is None:
            label = music_label
        else:
            label = song_label if end - start >= min_song_duration else jingle_label
        segments.append(Segment(start=start, end=end, label=label, score=score))
    return segments


def _word_start_and_text(word: Any) -> tuple[float, str] | None:
    """``(start, text)`` of a Scribe-style word dict or a ``Cue``-like object."""
    if isinstance(word, dict):
        if word.get("type", "word") not in ("word", "audio_event"):
            return None  # Scribe "spacing" tokens
        start, text = word.get("start"), word.get("text", "")
    else:
        start, text = getattr(word, "start", None), getattr(word, "text", "")
    text = " ".join(str(text).split())
    return (float(start), text) if start is not None and text else None


def text_before_segments(
    segments: Iterable[Segment | tuple[float, float]],
    words: Iterable[Any],
    *,
    before_seconds: float = DEFAULT_TEXT_BEFORE_SECONDS,
    after_seconds: float = DEFAULT_TEXT_AFTER_SECONDS,
) -> list[str]:
    """The transcript text just before each segment starts (one string per segment).

    Radio hosts announce a song right before it plays, so for each segment this
    joins the words that start within ``[start - before_seconds, start +
    after_seconds]``. It does not try to name the song; it hands you the text
    where the name usually is.

    Args:
        segments: ``Segment``s or ``(start, end)`` tuples (seconds); usually the
            songs from ``find_segments(..., strategy="audioset")``.
        words: Word-timestamped transcript items: Scribe-style dicts with
            ``"text"`` and ``"start"`` (``transcribe(...)["words"]``,
            :func:`mixing.srt.parse_vtt_words`), or ``Cue``-like objects with
            ``.text`` / ``.start`` (:func:`mixing.srt.parse_srt`).
        before_seconds: How far before a segment's start to look.
        after_seconds: How far past its start to keep looking (captions lag).

    >>> words = [{"text": "next", "start": 8.0}, {"text": "up,", "start": 8.5},
    ...          {"text": " ", "start": 8.9, "type": "spacing"},
    ...          {"text": "Yesterday", "start": 9.0}, {"text": "la", "start": 40.0}]
    >>> text_before_segments([Segment(10.0, 60.0, label="song")], words)
    ['next up, Yesterday']
    """
    items = [x for x in map(_word_start_and_text, words) if x is not None]
    out = []
    for seg in segments:
        start = float(seg.start if hasattr(seg, "start") else seg[0])
        lo, hi = start - before_seconds, start + after_seconds
        out.append(" ".join(text for t, text in items if lo <= t <= hi))
    return out


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------


def _merge_close_segments(segments: list[Segment], merge_gap: float) -> list[Segment]:
    """Merge consecutive segments whose gap is < merge_gap and whose label matches."""
    if not segments:
        return segments
    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        if s.start - prev.end <= merge_gap and s.label == prev.label:
            merged[-1] = Segment(
                start=prev.start,
                end=s.end,
                label=prev.label,
                score=max(
                    prev.score if prev.score is not None else float("-inf"),
                    s.score if s.score is not None else float("-inf"),
                )
                if (prev.score is not None or s.score is not None)
                else None,
            )
        else:
            merged.append(s)
    return merged


def _split_long_segments(segments: list[Segment], max_duration: float) -> list[Segment]:
    """Split any segment longer than ``max_duration`` into equal pieces."""
    out = []
    for s in segments:
        if s.duration <= max_duration:
            out.append(s)
            continue
        n_pieces = int(np.ceil(s.duration / max_duration))
        piece_dur = s.duration / n_pieces
        for k in range(n_pieces):
            out.append(
                Segment(
                    start=s.start + k * piece_dur,
                    end=s.start + (k + 1) * piece_dur,
                    label=s.label,
                    score=s.score,
                )
            )
    return out


def _absorb_short_segments(
    segments: list[Segment], min_duration: float
) -> list[Segment]:
    """Merge any segment shorter than ``min_duration`` into its longer neighbor."""
    if not segments:
        return segments
    work = list(segments)
    changed = True
    while changed and len(work) > 1:
        changed = False
        for i, s in enumerate(work):
            if s.duration >= min_duration:
                continue
            # Pick the neighbor to merge into.
            left = work[i - 1] if i > 0 else None
            right = work[i + 1] if i + 1 < len(work) else None
            if left is None and right is None:
                break
            if left is not None and (right is None or left.duration >= right.duration):
                merged = Segment(
                    start=left.start,
                    end=s.end,
                    label=left.label,
                    score=left.score,
                )
                work = work[: i - 1] + [merged] + work[i + 1 :]
            else:
                merged = Segment(
                    start=s.start,
                    end=right.end,
                    label=right.label,
                    score=right.score,
                )
                work = work[:i] + [merged] + work[i + 2 :]
            changed = True
            break
    return work


# ----------------------------------------------------------------------------
# Strategy registry and top-level entry points
# ----------------------------------------------------------------------------


SegmentationStrategy = Union[
    SegmentationStrategyName,
    Callable[..., list[Segment]],
]

_STRATEGIES: dict[str, Callable[..., list[Segment]]] = {
    "silence": segment_by_silence,
    "energy_novelty": segment_by_energy,
    "self_similarity": segment_by_self_similarity,
    "speech_music": segment_by_speech_music,
    "audioset": segment_by_audioset,
}


def _resolve_strategy(
    strategy: SegmentationStrategy,
) -> Callable[..., list[Segment]]:
    if callable(strategy):
        return strategy
    if strategy in _STRATEGIES:
        return _STRATEGIES[strategy]
    raise ValueError(
        f"Unknown segmentation strategy: {strategy!r}. "
        f"Choose one of {sorted(_STRATEGIES)} or pass a callable."
    )


def find_segments(
    audio: AudioSource,
    *,
    strategy: SegmentationStrategy = "silence",
    min_segment_duration: float = 0.0,
    max_segment_duration: float | None = None,
    merge_gap: float = 0.0,
    pad_start: float = 0.0,
    pad_end: float = 0.0,
    **strategy_kwargs,
) -> list[Segment]:
    """Find segment boundaries in ``audio`` using a chosen strategy.

    Args:
        audio: File path, numpy array, or pydub ``AudioSegment``.
        strategy: Strategy name (``"silence"``, ``"energy_novelty"``,
            ``"self_similarity"``, ``"speech_music"``, ``"audioset"``) or a
            callable ``(AudioSegment, **kwargs) -> list[Segment]``.
        min_segment_duration: Drop or merge segments shorter than this.
        max_segment_duration: Split segments longer than this into equal pieces.
        merge_gap: Merge consecutive same-label segments separated by less
            than this gap (seconds).
        pad_start: Extend each segment backwards by this many seconds.
        pad_end: Extend each segment forwards by this many seconds.
        **strategy_kwargs: Forwarded to the chosen strategy.

    Returns:
        List of ``Segment`` instances. Use ``.as_start_end()`` or
        ``.as_offset_duration()`` to get plain timestamp tuples.

    Examples:
        >>> from mixing.audio import find_segments  # doctest: +SKIP
        >>> segs = find_segments("mix.mp3", strategy="silence",
        ...                       silence_thresh_db=-45)  # doctest: +SKIP
        >>> [s.as_offset_duration() for s in segs]  # doctest: +SKIP
        [(0.0, 245.3), (247.1, 198.4), ...]
    """
    audio_seg = _normalize_audio_source(audio, target_type="AudioSegment")
    total_duration = len(audio_seg) / 1000.0

    fn = _resolve_strategy(strategy)
    segments = fn(audio_seg, **strategy_kwargs)

    if merge_gap > 0:
        segments = _merge_close_segments(segments, merge_gap)
    if min_segment_duration > 0:
        segments = _absorb_short_segments(segments, min_segment_duration)
    if max_segment_duration is not None and max_segment_duration > 0:
        segments = _split_long_segments(segments, max_segment_duration)

    if pad_start or pad_end:
        segments = [
            Segment(
                start=max(0.0, s.start - pad_start),
                end=min(total_duration, s.end + pad_end),
                label=s.label,
                score=s.score,
            )
            for s in segments
        ]
    return segments


def extract_segments(
    audio: AudioSource,
    segments: Sequence[Segment | tuple[float, float]] | None = None,
    *,
    output: str | Path | None = None,
    name_template: str = "{stem}_{idx:03d}{ext}",
    format: str = "mp3",
    bitrate: str = "192k",
    strategy: SegmentationStrategy = "silence",
    **strategy_kwargs,
) -> list[Path]:
    """Save each segment as its own audio file.

    If ``segments`` is ``None``, this calls :func:`find_segments` first using
    ``strategy`` and ``strategy_kwargs``.

    Args:
        audio: File path, numpy array, or pydub ``AudioSegment``.
        segments: Either a list of ``Segment`` objects, or a list of
            ``(start, end)`` tuples in seconds. If ``None``, segments are
            discovered automatically.
        output: Destination directory to write the per-segment files to
            (created if missing). This multi-file producer treats ``output``
            strictly as a directory — one file per segment is written into it.
            Defaults to the source file's parent, or the current directory
            if ``audio`` isn't a path.
        name_template: Filename template. Available fields: ``{stem}``,
            ``{idx}``, ``{label}``, ``{start}``, ``{end}``, ``{ext}``.
        format: Output audio format (e.g. ``"mp3"``, ``"wav"``).
        bitrate: Bitrate for compressed formats.
        strategy, **strategy_kwargs: Used only when ``segments`` is ``None``.

    Returns:
        List of paths to the saved files, in segment order.

    Examples:
        >>> from mixing.audio import extract_segments  # doctest: +SKIP
        >>> # Auto-detect and save in one call
        >>> paths = extract_segments(
        ...     "concert.wav", strategy="self_similarity",
        ...     output="songs/", format="mp3",
        ...     kernel_seconds=14.0,
        ... )  # doctest: +SKIP
        >>>
        >>> # Or pass timestamps you already have
        >>> paths = extract_segments(
        ...     "mix.mp3",
        ...     segments=[(0, 245), (247, 445), (445, 600)],
        ...     output="tracks/",
        ... )  # doctest: +SKIP
    """
    audio_seg = _normalize_audio_source(audio, target_type="AudioSegment")

    if segments is None:
        segs = find_segments(audio_seg, strategy=strategy, **strategy_kwargs)
    else:
        segs = [
            s if isinstance(s, Segment) else Segment(start=float(s[0]), end=float(s[1]))
            for s in segments
        ]

    # Resolve output directory and filename stem.
    if isinstance(audio, (str, Path)):
        src_path = Path(audio)
        stem = src_path.stem
        out_dir = Path(output) if output is not None else src_path.parent
    else:
        stem = "audio"
        out_dir = Path(output) if output is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = "." + format.lstrip(".")
    paths: list[Path] = []
    for idx, seg in enumerate(segs):
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        clip = audio_seg[start_ms:end_ms]
        filename = name_template.format(
            stem=stem,
            idx=idx,
            label=seg.label or "segment",
            start=int(seg.start),
            end=int(seg.end),
            ext=ext,
        )
        out_path = out_dir / filename
        clip.export(str(out_path), format=format, bitrate=bitrate)
        paths.append(out_path)
    return paths
