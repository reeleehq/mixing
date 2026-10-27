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
  speech vs music, etc.) using pluggable strategies; ``strategy="audioset"``
  (extra ``ml``) tells sung songs from talk on radio shows
- text_before_segments: transcript words just before each segment (song intros)
- loudness, loudness_range, true_peak_db, level_turns: BS.1770 LUFS meter, LRA, true peak; per-speaker-turn loudness
  matching (EQ, compress, gain to a target LUFS)
- process_turns: run a callable on each (padded) turn of a recording and splice
  the results back at exact length
- find_talk_edges: the host's talk at the start/end of a carved song (talk over
  music counts), with the music proper's clean start/end
- separate_speech, remove_talk: split speech from music (pluggable separators:
  MossFormer2-SE inverted by default, extra ``separation``; htdemucs for
  instrumental beds, extra ``demucs``), and remove a host's talk inside given
  spans while keeping the music (sample-identical outside them)
- voice_profile, shift_pitch: per-speaker median f0, loudness, spectral centroid
  and low-band (boom) ratio; pitch shift by semitones (ffmpeg rubberband with
  formant preservation, stock-ffmpeg resample, or librosa -- extra ``beats``)
- contains_song, landmarks: does this clip contain the song? A landmark-hash
  fingerprint verdict (``"contains"`` / ``"absent"`` / ``"undecided"``) with the
  clip's offset in the song; numpy + scipy only, thresholds uncalibrated

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
    segment_by_audioset,
    text_before_segments,
)
from .tagging import WindowProbabilities, ast_audioset_classifier
from .talk_edges import TalkEdges, find_talk_edges, talk_edges_from_probs
from .beats import beat_grid, BeatGrid
from .separation import (
    SeparationStrategy,
    SEPARATORS,
    DEFAULT_SEPARATOR,
    separate_speech,
    remove_talk,
    invert_enhancer,
)
from .leveling import (
    loudness,
    loudness_range,
    true_peak_db,
    level_turns,
    noise_floor_db,
)
from .turns import process_turns
from .fingerprint import (
    ABSENT,
    CONTAINS,
    SONG_VERDICTS,
    UNDECIDED,
    LandmarkParams,
    Landmarks,
    MatchParams,
    SongMatch,
    SongThresholds,
    SongVerdict,
    contains_song,
    landmarks,
    song_verdict,
)
from .voice import voice_profile, shift_pitch, has_rubberband, pyin_tracker, F0Tracker
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
    "segment_by_audioset",
    "text_before_segments",
    "loudness",
    "loudness_range",
    "true_peak_db",
    "level_turns",
    "noise_floor_db",
    "process_turns",
    "voice_profile",
    "pyin_tracker",
    "F0Tracker",
    "shift_pitch",
    "has_rubberband",
    "WindowProbabilities",
    "ast_audioset_classifier",
    "TalkEdges",
    "find_talk_edges",
    "talk_edges_from_probs",
    "beat_grid",
    "BeatGrid",
    "SeparationStrategy",
    "SEPARATORS",
    "DEFAULT_SEPARATOR",
    "separate_speech",
    "remove_talk",
    "invert_enhancer",
    "contains_song",
    "landmarks",
    "song_verdict",
    "SongMatch",
    "SongVerdict",
    "Landmarks",
    "LandmarkParams",
    "MatchParams",
    "SongThresholds",
    "CONTAINS",
    "ABSENT",
    "UNDECIDED",
    "SONG_VERDICTS",
]
