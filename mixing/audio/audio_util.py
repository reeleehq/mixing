"""Audio utility functions."""

import numpy as np
from pathlib import Path
from typing import Union
from ..util import require_package
from ._pcm import segment_samples


AudioSource = Union[str, Path, np.ndarray, "AudioSegment"]


def _normalize_audio_source(source: AudioSource, *, target_type: str = "AudioSegment"):
    """
    Convert various audio source types to a standard format.

    Args:
        source: Audio source (filepath, numpy array, or AudioSegment)
        target_type: Target type to convert to ("AudioSegment" or "ndarray")

    Returns:
        Audio in target format
    """
    AudioSegment = require_package("pydub").AudioSegment

    if target_type == "AudioSegment":
        if isinstance(source, AudioSegment):
            return source
        elif isinstance(source, (str, Path)):
            return AudioSegment.from_file(str(source))
        elif isinstance(source, np.ndarray):
            # Assume 16-bit PCM, mono or stereo
            # Convert to bytes
            audio_bytes = (source * 32767).astype(np.int16).tobytes()
            channels = 1 if source.ndim == 1 else source.shape[1]
            return AudioSegment(
                audio_bytes,
                frame_rate=44100,  # Default, may need parameterization
                sample_width=2,  # 16-bit
                channels=channels,
            )
        else:
            raise TypeError(f"Unsupported audio source type: {type(source).__name__}")
    elif target_type == "ndarray":
        if isinstance(source, np.ndarray):
            return source
        elif isinstance(source, (str, Path)):
            source = AudioSegment.from_file(str(source))
        if isinstance(source, AudioSegment):
            # interleaved, in [-1, 1] whatever the sample width
            return segment_samples(source, dtype=np.float32).ravel()
        else:
            raise TypeError(f"Unsupported audio source type: {type(source).__name__}")
    else:
        raise ValueError(f"Invalid target_type: {target_type}")


def get_audio_info(source: AudioSource) -> dict:
    """
    Get audio properties (duration, sample rate, channels).

    Args:
        source: Audio source (filepath, numpy array, or AudioSegment)

    Returns:
        Dictionary with audio properties

    Examples:
        >>> info = get_audio_info("audio.mp3")  # doctest: +SKIP
        >>> info['duration_seconds']  # doctest: +SKIP
        120.5
    """
    audio = _normalize_audio_source(source, target_type="AudioSegment")
    return {
        "duration_seconds": len(audio) / 1000.0,
        "duration_ms": len(audio),
        "sample_rate": audio.frame_rate,
        "channels": audio.channels,
        "sample_width": audio.sample_width,
        "frame_count": audio.frame_count(),
    }
