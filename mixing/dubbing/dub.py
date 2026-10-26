"""SRT-driven dubbing: replace a video's audio with TTS narration.

Given a video and an SRT transcript, synthesize each cue with an ElevenLabs
voice, lay the clips onto a silent track at their cue start times, and mux the
result back over the video — producing a re-voiced (or translated) version
that stays aligned to the original timeline.

Timing fit (``fit="speed"``, the default): when a synthesized line is longer
than its cue window, it is gently time-compressed (capped at ``max_speedup``)
so downstream cues keep their start times; when shorter, the slot is padded
with silence. This keeps lip/scene sync without clipping speech. Use
``fit="natural"`` to never alter speed (lines may drift later in the timeline).

Quick start:
    >>> from mixing.dubbing import dub_video_from_srt  # doctest: +SKIP
    >>> dub_video_from_srt(  # doctest: +SKIP
    ...     "promo.mp4", "promo.srt",
    ...     voice_id="<elevenlabs-voice-id>",
    ...     output="promo.en.mp4",
    ... )
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from mixing.dubbing.srt import Cue, parse_srt
from mixing.dubbing.tts import (
    DFLT_MODEL_ID,
    DFLT_OUTPUT_FORMAT,
    synthesize_to_file,
)
from mixing.video.video_ops import assemble_audio_track, replace_audio

PathLike = str | Path

#: A synthesizer maps (text, out_path) -> written path. Lets callers swap the
#: TTS backend (or inject a stub in tests) without touching the pipeline.
SynthFn = Callable[[str, Path], Path]


def dub_video_from_srt(
    video: PathLike,
    srt: PathLike | str | Sequence[Cue],
    *,
    voice_id: str,
    output: PathLike | None = None,
    api_key: str | None = None,
    model_id: str = DFLT_MODEL_ID,
    output_format: str = DFLT_OUTPUT_FORMAT,
    voice_settings: dict | None = None,
    language_code: str | None = None,
    fit: str = "speed",
    max_speedup: float = 1.5,
    keep_original_audio: float = 0.0,
    work_dir: PathLike | None = None,
    keep_work: bool = False,
    cache: bool = True,
    synth_fn: SynthFn | None = None,
) -> Path:
    """Replace ``video``'s audio with TTS narration built from ``srt``.

    Args:
        video: Path to the source video.
        srt: An ``.srt`` file path, raw SRT text, or a list of :class:`Cue`.
        voice_id: ElevenLabs voice id for the narration.
        output: Output video path. Defaults to
            ``<video-stem>.<language_code or 'dub'>.mp4`` next to the source.
        api_key: ElevenLabs API key (falls back to ``ELEVENLABS_API_KEY`` or ``ELEVEN_API_KEY``).
        model_id: TTS model id (default ``eleven_multilingual_v2``).
        output_format: TTS audio format for the per-cue clips.
        voice_settings: Override default ElevenLabs voice settings.
        language_code: Optional ISO-639-1 hint forwarded to the TTS model and
            used in the default output filename.
        fit: ``"speed"`` (default) time-compresses overlong lines (capped at
            ``max_speedup``) to hold sync; ``"natural"`` never changes speed.
        max_speedup: Maximum tempo factor when ``fit="speed"`` (clamped to
            ffmpeg's single-``atempo`` ceiling of 2.0).
        keep_original_audio: Fraction (0..1) of the original audio to keep
            mixed under the narration. ``0.0`` (default) replaces it entirely;
            e.g. ``0.15`` keeps a little background music/ambience.
        work_dir: Directory for intermediate audio. Defaults to a temp dir.
        keep_work: Keep the intermediate per-cue clips and assembled track.
        cache: Cache TTS calls on disk (skips re-synthesis of identical lines).
        synth_fn: Override the synthesizer ``(text, out_path) -> path``. When
            ``None``, uses ElevenLabs via :func:`mixing.dubbing.tts.synthesize_to_file`.

    Returns:
        Path to the dubbed video.

    Raises:
        ValueError: ``fit`` is not a recognized strategy.
    """
    if fit not in ("speed", "natural"):
        raise ValueError(f"fit must be 'speed' or 'natural', got {fit!r}")

    video = Path(video)
    cues = _load_cues(srt)
    out_path = (
        Path(output)
        if output is not None
        else video.with_suffix("").with_name(
            f"{video.stem}.{language_code or 'dub'}.mp4"
        )
    )

    if synth_fn is None:

        def synth_fn(text: str, dest: Path) -> Path:  # noqa: E306
            return synthesize_to_file(
                text,
                voice_id,
                dest,
                api_key=api_key,
                model_id=model_id,
                output_format=output_format,
                voice_settings=voice_settings,
                language_code=language_code,
                cache=cache,
            )

    owns_work_dir = work_dir is None
    work = (
        Path(work_dir)
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="mixing_dub_"))
    )
    work.mkdir(parents=True, exist_ok=True)
    try:
        video_dur = _media_duration(video)
        segments = _build_segments(
            cues,
            work=work,
            synth_fn=synth_fn,
            fit=fit,
            max_speedup=max_speedup,
            video_dur=video_dur,
        )
        track_path = work / "narration_track.wav"
        assemble_audio_track(segments, output=track_path)
        replace_audio(
            str(video),
            str(track_path),
            mix_ratio=1.0 - keep_original_audio,
            output=str(out_path),
            match_duration=False,
        )
    finally:
        if owns_work_dir and not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return out_path


def _load_cues(srt: PathLike | str | Sequence[Cue]) -> list[Cue]:
    if isinstance(srt, (list, tuple)):
        return list(srt)
    text = str(srt)
    # Distinguish an SRT *path* from raw SRT *text*.
    if "-->" not in text:
        text = Path(text).read_text()
    return parse_srt(text)


def _build_segments(
    cues: Sequence[Cue],
    *,
    work: Path,
    synth_fn: SynthFn,
    fit: str,
    max_speedup: float,
    video_dur: float,
) -> list[tuple[Path | None, float]]:
    """Synthesize each cue and lay clips on an absolute timeline with gaps.

    Returns ``(audio_path_or_None, seg_seconds)`` segments for
    :func:`assemble_audio_track`: a silence gap before each cue (so it starts
    at its cue time) followed by the cue's clip. Each clip's *available
    window* runs from its start to the **next cue's start** — i.e. it may use
    the natural pause after the line, not just the tight cue ``end``. Only
    when a clip still overruns that window is it time-compressed (``fit=
    "speed"``, capped at ``max_speedup``), which keeps the read unhurried
    wherever the original left room. No speech is ever trimmed.
    """
    cap = min(max_speedup, 2.0)  # single ffmpeg atempo handles up to 2.0
    ordered = sorted([c for c in cues if c.text.strip()], key=lambda c: c.start)
    segments: list[tuple[Path | None, float]] = []
    cursor = 0.0
    for i, cue in enumerate(ordered):
        gap = cue.start - cursor
        if gap > 1e-3:
            segments.append((None, gap))
            cursor += gap

        clip = synth_fn(cue.text, work / f"cue_{i:04d}.mp3")
        clip_dur = _media_duration(clip)

        # Window the line can occupy before the next line must begin (or the
        # end of the video for the last line). Uses the gap after the cue too.
        next_start = ordered[i + 1].start if i + 1 < len(ordered) else video_dur
        window = max(0.0, next_start - cursor)

        if fit == "speed" and window > 0.05 and clip_dur > window:
            factor = min(clip_dur / window, cap)
            if factor > 1.001:
                sped = work / f"cue_{i:04d}.spd.wav"
                _atempo(clip, sped, factor)
                clip, clip_dur = sped, _media_duration(sped)

        segments.append((clip, clip_dur))
        cursor += clip_dur

    if video_dur - cursor > 1e-3:
        segments.append((None, video_dur - cursor))
    return segments


def _media_duration(path: PathLike) -> float:
    """Duration of a media file in seconds (via ffprobe, moviepy fallback)."""
    if shutil.which("ffprobe"):
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        try:
            return float(out.stdout.strip())
        except ValueError:
            pass
    import moviepy as mp  # fallback; pulls the whole clip

    with mp.AudioFileClip(str(path)) as clip:
        return float(clip.duration)


def _atempo(in_path: PathLike, out_path: PathLike, factor: float) -> Path:
    """Time-stretch audio by ``factor`` (speed up if >1) via ffmpeg ``atempo``."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-filter:a",
            f"atempo={factor:.5f}",
            "-vn",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return Path(out_path)
