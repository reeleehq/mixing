"""High-level filler-removal pipeline.

Orchestrates: extract audio -> Scribe transcribe -> filler detection ->
ffmpeg cut -> write transcript outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Union

from mixing.transcript.fillers import (
    DEFAULT_AUDIO_EVENTS_TO_CUT,
    DEFAULT_FILLER_TOKENS,
    build_cuts,
    keeps_from_cuts,
)
from mixing.transcript.formats import (
    words_to_prose,
    words_to_srt,
    words_to_srt_remapped,
)
from mixing.transcript.media import apply_keeps, extract_audio
from mixing.transcript.scribe import transcribe

PathLike = Union[str, Path]


@dataclass
class FillerRemovalResult:
    """Paths and computed ranges produced by :func:`remove_fillers`."""

    cleaned_media: Path
    transcript_md: Path
    transcript_srt: Path
    cleaned_md: Path
    cleaned_srt: Path
    scribe_json: Path
    cuts_json: Path
    keeps_json: Path
    duration: float
    cuts: list[dict] = field(default_factory=list)
    keeps: list[dict] = field(default_factory=list)


def remove_fillers(
    input_media: PathLike,
    output_dir: PathLike,
    *,
    output_media: PathLike | None = None,
    fillers: Iterable[str] = DEFAULT_FILLER_TOKENS,
    audio_events: Iterable[str] = DEFAULT_AUDIO_EVENTS_TO_CUT,
    api_key: str | None = None,
    scribe_kwargs: Mapping[str, object] | None = None,
    apply_keeps_kwargs: Mapping[str, object] | None = None,
    keep_intermediate: bool = True,
    scribe_data: dict | None = None,
) -> FillerRemovalResult:
    """End-to-end filler removal.

    Args:
        input_media: Video or audio file to clean.
        output_dir: Directory to write transcripts and intermediate files into.
            Created if missing.
        output_media: Output cleaned media path. Defaults to
            ``output_dir / f"{stem}.cleaned.mp4"``.
        fillers / audio_events: Override the default filler / event sets.
        api_key: ElevenLabs API key (else env ``ELEVENLABS_API_KEY`` or ``ELEVEN_API_KEY``).
        scribe_kwargs: Extra kwargs forwarded to :func:`transcribe`.
        apply_keeps_kwargs: Extra kwargs forwarded to :func:`apply_keeps`.
        keep_intermediate: If ``False``, delete the extracted audio file.
        scribe_data: If provided, skip the network call and use this dict
            (must match Scribe's response shape). Useful for tests / replays.

    Returns:
        A :class:`FillerRemovalResult` with file paths and the computed
        cut/keep ranges.
    """
    input_media = Path(input_media)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_media is None:
        output_media = output_dir / f"{input_media.stem}.cleaned.mp4"

    if scribe_data is None:
        audio_path = output_dir / f"{input_media.stem}.mp3"
        extract_audio(input_media, audio_path)
        scribe_kw = dict(scribe_kwargs or {})
        if api_key is not None:
            scribe_kw["api_key"] = api_key
        scribe_data = transcribe(audio_path, **scribe_kw)
        if not keep_intermediate:
            audio_path.unlink(missing_ok=True)

    words = scribe_data["words"]
    duration = float(scribe_data["audio_duration_secs"])
    cuts = build_cuts(words, fillers=fillers, audio_events=audio_events)
    keeps = keeps_from_cuts(cuts, duration)

    apply_keeps_kw = dict(apply_keeps_kwargs or {})
    apply_keeps(input_media, output_media, keeps, **apply_keeps_kw)

    paths = _write_outputs(
        output_dir=output_dir,
        scribe_data=scribe_data,
        words=words,
        cuts=cuts,
        keeps=keeps,
        fillers=fillers,
        audio_events=audio_events,
    )

    return FillerRemovalResult(
        cleaned_media=Path(output_media),
        duration=duration,
        cuts=cuts,
        keeps=keeps,
        **paths,
    )


def _write_outputs(
    *,
    output_dir: Path,
    scribe_data: dict,
    words: list,
    cuts: list,
    keeps: list,
    fillers: Iterable[str],
    audio_events: Iterable[str],
) -> dict:
    scribe_json = output_dir / "scribe.json"
    cuts_json = output_dir / "cuts.json"
    keeps_json = output_dir / "keeps.json"
    transcript_md = output_dir / "transcript.md"
    transcript_srt = output_dir / "transcript.srt"
    cleaned_md = output_dir / "transcript.cleaned.md"
    cleaned_srt = output_dir / "transcript.cleaned.srt"

    scribe_json.write_text(json.dumps(scribe_data, indent=2))
    cuts_json.write_text(json.dumps(cuts, indent=2))
    keeps_json.write_text(json.dumps(keeps, indent=2))
    transcript_md.write_text(scribe_data.get("text", ""))
    transcript_srt.write_text(words_to_srt(words))
    cleaned_md.write_text(
        words_to_prose(
            words, drop_fillers=True, fillers=fillers, audio_events=audio_events
        )
    )
    cleaned_srt.write_text(
        words_to_srt_remapped(words, cuts, fillers=fillers, audio_events=audio_events)
    )
    return {
        "scribe_json": scribe_json,
        "cuts_json": cuts_json,
        "keeps_json": keeps_json,
        "transcript_md": transcript_md,
        "transcript_srt": transcript_srt,
        "cleaned_md": cleaned_md,
        "cleaned_srt": cleaned_srt,
    }
