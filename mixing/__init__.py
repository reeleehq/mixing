"""Tools for video and audio editing.

``mixing`` is organized into focused subpackages, each with a clear dependency
footprint so you only pay for what you import:

- :mod:`mixing.audio` — sliceable :class:`~mixing.audio.Audio`, fades, crop,
  concat, overlay, alignment, segmentation (needs ``pydub``).
- :mod:`mixing.video` — sliceable :class:`~mixing.video.Video`, crop, loop,
  audio replace/normalize, Ken Burns, thumbnails, subtitles (needs ``moviepy``
  / ``opencv``).
- :mod:`mixing.transcript` — speech-to-text (ElevenLabs Scribe, stdlib HTTP),
  filler removal, transcript formats (no heavy deps).
- :mod:`mixing.dubbing` — text-to-speech re-voicing / translation from SRT.
- :mod:`mixing.srt` — canonical SRT/timeline parsing & formatting (pure).
- :mod:`mixing.chapters` — transcript → chapter markers (pure; LLM optional).

**Lazy by design.** Importing ``mixing`` (or a light submodule such as
``mixing.chapters`` / ``mixing.srt``) does **not** import ``moviepy`` or
``opencv``. The heavy backends load only when you first touch a name that needs
them — e.g. ``mixing.Video`` or ``from mixing.video import replace_audio``.

The top-level namespace is a curated, lazily-resolved facade over the
subpackages; see :data:`__all__`. For the full surface of a subsystem, import
it directly (``from mixing.audio import ...``).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

# --- Eager, dependency-light exports -------------------------------------
# These pull no heavy media backends, keeping ``import mixing`` (and
# ``import mixing.chapters``) cheap.
from mixing.chapters import Chapter, detect_chapters
from mixing.util import has_ffmpeg

# Library convention: mixing never configures logging or writes to stdout on
# its own. A ``NullHandler`` on the package root means every ``mixing.*``
# logger is silent until a consuming application adds its own handler — the
# single package-wide seam for what was ad hoc ``print()``/``quiet=`` noise
# (see #32, #39).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# --- Lazy facade ----------------------------------------------------------
# name -> the submodule that defines it. The submodule (and its backend deps)
# is imported only when the name is first accessed via ``__getattr__``.
_LAZY: dict[str, str] = {
    # video (moviepy / opencv)
    "Video": "mixing.video",
    "VideoFrames": "mixing.video",
    "crop_video": "mixing.video",
    "loop_video": "mixing.video",
    "change_speed": "mixing.video",
    "replace_audio": "mixing.video",
    "normalize_audio": "mixing.video",
    "overlay_ambient_bed": "mixing.video",
    "save_frame": "mixing.video",
    "ken_burns_video": "mixing.video",
    "ken_burns_film": "mixing.video",
    "assemble_audio_track": "mixing.video",
    "concatenate_videos": "mixing.video",
    "shots": "mixing.video",
    "cutaways": "mixing.video",
    "plan_cutaways": "mixing.video",
    "Cutaway": "mixing.video",
    # video transitions (transform_clips-shaped: list[clip] -> Iterable[clip])
    "crossfade_transition": "mixing.video",
    # …and how a transition declares the overlapped join it needs
    "needs_crossfade_overlap": "mixing.video",
    "crossfade_overlap": "mixing.video",
    # …and what the clips can actually carry (past it, clips get dropped)
    "max_overlap_for_clips": "mixing.video",
    "fade_through_black": "mixing.video",
    "slow_motion_blend": "mixing.video",
    "trim_and_crossfade": "mixing.video",
    "overlap_blend": "mixing.video",
    "trim_first_frame_from_subsequent_clips": "mixing.video",
    "make_thumbnail": "mixing.video",
    "make_gif": "mixing.video",
    "write_subtitles_in_video": "mixing.video",
    "get_video_dimensions": "mixing.video",
    "resize_to_dimensions": "mixing.video",
    "normalize_video_dimensions": "mixing.video",
    # audio (pydub)
    "Audio": "mixing.audio",
    "AudioSamples": "mixing.audio",
    "fade_in": "mixing.audio",
    "fade_out": "mixing.audio",
    "crop_audio": "mixing.audio",
    "concatenate_audio": "mixing.audio",
    "overlay_audio": "mixing.audio",
    "loop_audio": "mixing.audio",
    "duck_audio": "mixing.audio",
    "save_audio_clip": "mixing.audio",
    "find_audio_offset": "mixing.audio",
    "Segment": "mixing.audio",
    "find_segments": "mixing.audio",
    "extract_segments": "mixing.audio",
    # transcript (stdlib HTTP)
    "transcribe": "mixing.transcript",
    "remove_fillers": "mixing.transcript",
    "srt_for_media": "mixing.transcript",
    "FillerRemovalResult": "mixing.transcript",
    # dubbing (stdlib HTTP)
    "dub_video_from_srt": "mixing.dubbing",
    "text_to_speech": "mixing.dubbing",
    "translate_srt": "mixing.dubbing",
    # typed errors (pure)
    "MixingError": "mixing.errors",
    "WindowTooWideForClip": "mixing.errors",
    # srt / timeline (pure)
    "Cue": "mixing.srt",
    "parse_srt": "mixing.srt",
    "dump_srt": "mixing.srt",
    "srt_time_to_seconds": "mixing.srt",
    "seconds_to_srt_time": "mixing.srt",
}

# Submodules reachable as attributes after ``import mixing``.
_SUBMODULES = (
    "audio",
    "video",
    "transcript",
    "dubbing",
    "chapters",
    "srt",
    "egress",
    "errors",
    "util",
)

if TYPE_CHECKING:  # help static analyzers see the lazy names
    from mixing.audio import (  # noqa: F401
        Audio,
        AudioSamples,
        Segment,
        concatenate_audio,
        crop_audio,
        extract_segments,
        fade_in,
        fade_out,
        find_audio_offset,
        find_segments,
        overlay_audio,
        loop_audio,
        duck_audio,
        save_audio_clip,
    )
    from mixing.video import (  # noqa: F401
        Video,
        VideoFrames,
        assemble_audio_track,
        change_speed,
        concatenate_videos,
        crop_video,
        crossfade_overlap,
        crossfade_transition,
        fade_through_black,
        max_overlap_for_clips,
        needs_crossfade_overlap,
        overlap_blend,
        slow_motion_blend,
        trim_and_crossfade,
        trim_first_frame_from_subsequent_clips,
        get_video_dimensions,
        ken_burns_film,
        ken_burns_video,
        loop_video,
        make_gif,
        make_thumbnail,
        normalize_audio,
        normalize_video_dimensions,
        overlay_ambient_bed,
        replace_audio,
        shots,
        Cutaway,
        cutaways,
        plan_cutaways,
        resize_to_dimensions,
        save_frame,
        write_subtitles_in_video,
    )
    from mixing.transcript import (  # noqa: F401
        FillerRemovalResult,
        remove_fillers,
        srt_for_media,
        transcribe,
    )
    from mixing.dubbing import dub_video_from_srt, text_to_speech, translate_srt  # noqa: F401
    from mixing.errors import MixingError, WindowTooWideForClip  # noqa: F401


def __getattr__(name: str):
    """Resolve a facade name or submodule lazily (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is not None:
        value = getattr(importlib.import_module(module_path), name)
        globals()[name] = value  # cache so __getattr__ isn't hit again
        return value
    if name in _SUBMODULES:
        module = importlib.import_module(f"mixing.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))


__all__ = sorted(
    set(_LAZY) | {"Chapter", "detect_chapters", "has_ffmpeg"} | set(_SUBMODULES)
)
