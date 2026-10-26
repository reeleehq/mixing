"""Transcript-driven media editing.

Speech-to-text via ElevenLabs Scribe, filler-word detection, and
timeline-aware cuts of audio/video. Produces editable transcripts
(plain prose, SRT, word-level JSON) and applies cuts via ffmpeg.

The HTTP client uses stdlib only (no `requests` or `elevenlabs` SDK
dependency), so the package adds no required deps. The runtime needs
ffmpeg on PATH for media operations, and an ElevenLabs API key (env
var `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`) or explicit `api_key=`) for transcription.

Quick start:
    >>> from mixing.transcript import remove_fillers  # doctest: +SKIP
    >>> result = remove_fillers("input.mov", "out/")  # doctest: +SKIP
    >>> print(result.cleaned_media)  # doctest: +SKIP

Lower-level building blocks:
    >>> from mixing.transcript import (  # doctest: +SKIP
    ...     transcribe, build_cuts, keeps_from_cuts,
    ...     words_to_srt, words_to_prose, apply_keeps,
    ... )
"""

from mixing.transcript.scribe import (
    ELEVENLABS_STT_URL,
    default_cache_dir,
    transcribe,
)
from mixing.transcript.fillers import (
    DEFAULT_FILLER_TOKENS,
    DEFAULT_AUDIO_EVENTS_TO_CUT,
    is_filler,
    build_cuts,
    keeps_from_cuts,
    normalize_token,
)
from mixing.transcript.formats import (
    fmt_srt_time,
    words_to_srt,
    words_to_srt_remapped,
    words_to_prose,
    remap_time_after_cuts,
)
from mixing.transcript.media import extract_audio, apply_keeps
from mixing.transcript.persist import srt_for_media
from mixing.transcript.pipeline import remove_fillers, FillerRemovalResult

__all__ = [
    "transcribe",
    "default_cache_dir",
    "ELEVENLABS_STT_URL",
    "DEFAULT_FILLER_TOKENS",
    "DEFAULT_AUDIO_EVENTS_TO_CUT",
    "is_filler",
    "build_cuts",
    "keeps_from_cuts",
    "normalize_token",
    "fmt_srt_time",
    "words_to_srt",
    "words_to_srt_remapped",
    "words_to_prose",
    "remap_time_after_cuts",
    "extract_audio",
    "apply_keeps",
    "srt_for_media",
    "remove_fillers",
    "FillerRemovalResult",
]
