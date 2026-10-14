"""Audio mixing and editing functionality.

Main exports:
- Audio: Sliceable audio interface with time-based operations
- fade_in, fade_out: Apply fade effects
- crop_audio: Trim audio segments
- concatenate_audio: Join multiple audio files
- overlay_audio: Mix/overlay audio tracks
- loop_audio: Tile a clip (crossfaded) to a target duration
- duck_audio: Sidechain-duck a bed under a dialogue track
- save_audio_clip: Extract and save audio segments
- find_audio_offset: Align two recordings via cross-correlation
- find_segments, extract_segments: Split a long audio into pieces (songs,
  speech vs music, etc.) using pluggable strategies

Examples:
    >>> from mixing.audio import Audio, fade_in, concatenate_audio  # doctest: +SKIP
    >>> audio = Audio("song.mp3")  # doctest: +SKIP
    >>> segment = audio[10:30]  # 10s to 30s  # doctest: +SKIP
    >>> segment.save("clip.mp3")  # doctest: +SKIP

    >>> faded = fade_in("intro.mp3", duration=2.0)  # doctest: +SKIP
    >>> combined = concatenate_audio("part1.mp3", "part2.mp3", "part3.mp3")  # doctest: +SKIP

    >>> from mixing.audio import find_segments, extract_segments  # doctest: +SKIP
    >>> segs = find_segments("concert.wav", strategy="self_similarity")  # doctest: +SKIP
    >>> paths = extract_segments("concert.wav", segs, output="songs/")  # doctest: +SKIP
"""

from ..util import AUDIO_IMPORT_NAMES, require_media_extra

require_media_extra("mixing.audio", modules=AUDIO_IMPORT_NAMES)

from .audio_ops import (
    Audio,
    AudioSamples,
    fade_in,
    fade_out,
    crop_audio,
    concatenate_audio,
    overlay_audio,
    loop_audio,
    duck_audio,
    DEFAULT_LOOP_CROSSFADE_S,
    DEFAULT_DUCK_DB,
    DEFAULT_DUCK_THRESHOLD_DB,
    DEFAULT_DUCK_ATTACK_S,
    DEFAULT_DUCK_RELEASE_S,
    DEFAULT_DUCK_HOLD_S,
    DEFAULT_DUCK_FRAME_S,
    save_audio_clip,
    find_audio_offset,
    find_audio_offset_detailed,
    AudioOffset,
    align_clips_to_reference,
    aligned_spans,
    AlignedSpan,
    ClipAlignment,
    onset_envelope,
    ALIGNMENT_FEATURES,
)
from .segmentation import (
    Segment,
    find_segments,
    extract_segments,
    segment_by_silence,
    segment_by_energy,
    segment_by_self_similarity,
    segment_by_speech_music,
)
from .beats import beat_grid, BeatGrid
from ..errors import MixingError, WindowTooWideForClip

__all__ = [
    "Audio",
    "AudioSamples",
    "fade_in",
    "fade_out",
    "crop_audio",
    "concatenate_audio",
    "overlay_audio",
    "loop_audio",
    "duck_audio",
    "DEFAULT_LOOP_CROSSFADE_S",
    "DEFAULT_DUCK_DB",
    "DEFAULT_DUCK_THRESHOLD_DB",
    "DEFAULT_DUCK_ATTACK_S",
    "DEFAULT_DUCK_RELEASE_S",
    "DEFAULT_DUCK_HOLD_S",
    "DEFAULT_DUCK_FRAME_S",
    "save_audio_clip",
    "find_audio_offset",
    "find_audio_offset_detailed",
    "AudioOffset",
    "align_clips_to_reference",
    "aligned_spans",
    "AlignedSpan",
    "ClipAlignment",
    "onset_envelope",
    "ALIGNMENT_FEATURES",
    "MixingError",
    "WindowTooWideForClip",
    "Segment",
    "find_segments",
    "extract_segments",
    "segment_by_silence",
    "segment_by_energy",
    "segment_by_self_similarity",
    "segment_by_speech_music",
    "beat_grid",
    "BeatGrid",
]
