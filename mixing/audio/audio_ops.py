"""
Audio editing via slicing interface.

Provides lazy views into audio segments and comprehensive audio editing operations.

This module provides:
- `Audio`: A sliceable audio interface using `audio[start:end]` notation
- `fade_in()`, `fade_out()`: Apply fade effects
- `crop_audio()`: Trim audio segments
- `concatenate_audio()`: Join multiple audio files
- `overlay_audio()`: Mix audio tracks
- Flexible time units: seconds, samples, or milliseconds
- Integration with pydub for audio processing
- Clipboard support for audio file paths

Examples:
    >>> audio = Audio("my_audio.mp3")  # doctest: +SKIP
    >>> segment = audio[10:20]  # Lazy view, no copying  # doctest: +SKIP
    >>> segment.save("clip.mp3")  # Only then does it process  # doctest: +SKIP

    >>> # Apply fade effects
    >>> faded = fade_in(audio, duration=2.0)  # 2 second fade in  # doctest: +SKIP
    >>> faded.save("faded.mp3")  # doctest: +SKIP

    >>> # Concatenate audio files
    >>> combined = concatenate_audio(["intro.mp3", "main.mp3", "outro.mp3"])  # doctest: +SKIP
    >>> combined.save("full.mp3")  # doctest: +SKIP

    >>> # Overlay/mix audio
    >>> mixed = overlay_audio("background.mp3", "voice.mp3", position=5.0)  # doctest: +SKIP

Design principles:
- Lazy evaluation: Operations create views, not copies
- Facade pattern: Clean interface over pydub complexity
- Standard library interfaces: Uses Python's slice notation
- Dependency injection: Configurable time units and formats
- Open-closed: Extensible via keyword arguments
"""

from typing import Union, TYPE_CHECKING
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import io
import logging
import os
import tempfile
import numpy as np

from ..util import require_package, AudioTimeUnit, to_seconds, get_path_from_clipboard
from ._pcm import resample, segment_samples
from .audio_util import AudioSource, _normalize_audio_source
from ..egress import Output, deliver, is_sink, resolve_output_path
from ..errors import WindowTooWideForClip

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydub import AudioSegment


class AudioSamples(Mapping[int, float]):
    """
    Mapping interface to access audio samples by index.

    Provides dictionary-like access to audio samples with support for negative
    indexing and slicing. Samples are returned as normalized float values.

    Args:
        audio_src: Path to audio file or AudioSegment
        start_sample: Starting sample index (for segments)
        end_sample: Ending sample index (for segments)

    Examples:
        >>> audio_samples = AudioSamples("test_audio.mp3")  # doctest: +SKIP
        >>> sample = audio_samples[0]  # Get first sample  # doctest: +SKIP
        >>> last_sample = audio_samples[-1]  # Get last sample  # doctest: +SKIP
        >>> samples = list(audio_samples[1000:2000])  # Get samples 1000-1999  # doctest: +SKIP
    """

    def __init__(
        self,
        audio_src: Union[str, "os.PathLike[str]", "AudioSegment"],
        start_sample: int = 0,
        end_sample: int | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(audio_src, (str, os.PathLike)):
            # Accept both ``str`` and ``pathlib.Path`` / any os.PathLike
            audio_src = os.fspath(audio_src)
            self.audio_src = audio_src
            self._audio = AudioSegment.from_file(audio_src)
        else:
            self.audio_src = None
            self._audio = audio_src

        # Float samples in [-1, 1]: (n,) for mono, (n, channels) otherwise
        samples = segment_samples(self._audio, dtype=np.float32)
        self._samples = samples[:, 0] if self._audio.channels == 1 else samples

        self.start_sample = start_sample
        self.end_sample = end_sample if end_sample is not None else len(self._samples)
        self._sample_count = self.end_sample - self.start_sample

    def __len__(self) -> int:
        """Return number of samples in this view."""
        return self._sample_count

    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices in this view."""
        return iter(range(self.start_sample, self.end_sample))

    def __getitem__(self, key: int | slice) -> float | np.ndarray:
        """
        Get sample(s) by index or slice.

        Args:
            key: Integer index or slice object (relative to this view)

        Returns:
            Single sample value or array of samples
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(self._sample_count)
            abs_start = self.start_sample + start
            abs_stop = self.start_sample + stop
            return self._samples[abs_start:abs_stop:step]
        elif isinstance(key, int):
            if key < 0:
                key = self._sample_count + key
            if key < 0 or key >= self._sample_count:
                raise IndexError(
                    f"Sample index {key} out of range [0, {self._sample_count})"
                )
            abs_idx = self.start_sample + key
            return self._samples[abs_idx]
        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )


class Audio:
    """
    Sliceable interface for audio supporting time-based operations.

    Provides lazy views into audio segments using slice notation. Slicing returns
    new Audio instances (not copies), enabling chained operations.

    Args:
        src_path: Path to source audio file or AudioSegment
        time_unit: Unit for slice indices ('seconds', 'samples', 'milliseconds')
        start_time: Start time in seconds (for creating sub-views)
        end_time: End time in seconds (for creating sub-views)

    Examples:
        >>> audio = Audio("song.mp3")  # doctest: +SKIP
        >>>
        >>> # Get segment from 10s to 20s (returns Audio)
        >>> segment = audio[10:20]  # doctest: +SKIP
        >>> segment.save("clip.mp3")  # doctest: +SKIP
        >>>
        >>> # Use sample numbers as unit
        >>> audio_samples = Audio("song.mp3", time_unit="samples")  # doctest: +SKIP
        >>> segment = audio_samples[44100:88200]  # 1 second at 44.1kHz  # doctest: +SKIP
        >>>
        >>> # Get last 30 seconds
        >>> ending = audio[-30:]  # doctest: +SKIP
        >>>
        >>> # Chain operations
        >>> trimmed = audio[5:120]  # Trim to 5s-120s  # doctest: +SKIP
        >>> faded = trimmed.fade_in(2).fade_out(3)  # Apply fades  # doctest: +SKIP
        >>> faded.save("final.mp3")  # doctest: +SKIP
    """

    def __init__(
        self,
        src_path: Union[str, "os.PathLike[str]", "AudioSegment"],
        *,
        time_unit: AudioTimeUnit = "seconds",
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        AudioSegment = require_package("pydub").AudioSegment

        if isinstance(src_path, (str, os.PathLike)):
            # Accept both ``str`` and ``pathlib.Path`` / any os.PathLike
            src_path = os.fspath(src_path)
            self.src_path = str(src_path)
            self._audio = AudioSegment.from_file(src_path)
        else:
            self.src_path = None
            self._audio = src_path

        self.time_unit = time_unit
        self._start_time = start_time  # None means start of audio
        self._end_time = end_time  # None means end of audio

    @property
    def start_time(self) -> float:
        """Start time in seconds (0.0 if not set)."""
        return self._start_time if self._start_time is not None else 0.0

    @property
    def end_time(self) -> float:
        """End time in seconds (audio duration if not set)."""
        return self._end_time if self._end_time is not None else self.full_duration

    @property
    def full_duration(self) -> float:
        """Duration of the source audio in seconds, exact to the frame (not
        pydub's ``len``, which rounds to a millisecond)."""
        return self._audio.frame_count() / self._audio.frame_rate

    @property
    def duration(self) -> float:
        """Duration of this audio/segment in seconds."""
        return self.end_time - self.start_time

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._audio.frame_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._audio.channels

    @property
    def sample_count(self) -> int:
        """Number of frames (samples per channel) in this segment."""
        start, end = self._frame_bounds()
        return end - start

    def _frame_bounds(self) -> tuple[int, int]:
        """``(start, end)`` frame indices of this segment: each bound is the
        nearest frame to its time, clamped to the source; an open end is the
        last frame."""
        n = int(self._audio.frame_count())

        def frame(seconds: float) -> int:
            return min(n, max(0, round(seconds * self._audio.frame_rate)))

        end = n if self._end_time is None else frame(self._end_time)
        return frame(self.start_time), end

    def _normalize_index(self, idx: int | float | None, is_start: bool) -> float:
        """Convert slice index to seconds, handling None and negative indices."""
        if idx is None:
            return self.start_time if is_start else self.end_time

        # Convert to seconds based on time_unit
        idx_seconds = to_seconds(idx, unit=self.time_unit, rate=self.sample_rate)

        # Handle negative indices (from end of this segment)
        if idx_seconds < 0:
            idx_seconds = self.end_time + idx_seconds
        else:
            # Positive indices are relative to segment start
            idx_seconds = self.start_time + idx_seconds

        # Clamp to segment's valid range
        return max(self.start_time, min(idx_seconds, self.end_time))

    def __getitem__(self, key: int | float | slice) -> "Audio":
        """
        Get an audio segment using slice notation.

        Args:
            key: Slice for time range

        Returns:
            New Audio instance representing the segment

        Examples:
            >>> audio = Audio("test.mp3")  # doctest: +SKIP
            >>> segment = audio[10:20]  # 10s to 20s  # doctest: +SKIP
            >>> ending = audio[-30:]  # Last 30 seconds  # doctest: +SKIP
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported for audio slicing")

            start = self._normalize_index(key.start, is_start=True)
            end = self._normalize_index(key.stop, is_start=False)

            if start >= end:
                raise ValueError(
                    f"Invalid time range: start ({start}s) must be before end ({end}s)"
                )

            # An open end stays open (to the last frame, not the ms-rounded
            # duration)
            open_end = key.stop is None and self._end_time is None
            return Audio(
                self._audio,
                time_unit=self.time_unit,
                start_time=start,
                end_time=None if open_end else end,
            )

        elif isinstance(key, (int, float)):
            # Single sample/time point - return very short segment
            idx_seconds = to_seconds(key, unit=self.time_unit, rate=self.sample_rate)
            if idx_seconds < 0:
                time_seconds = self.end_time + idx_seconds
            else:
                time_seconds = self.start_time + idx_seconds

            time_seconds = max(self.start_time, min(time_seconds, self.end_time))
            sample_duration = 1.0 / self.sample_rate
            return Audio(
                self._audio,
                time_unit=self.time_unit,
                start_time=time_seconds,
                end_time=time_seconds + sample_duration,
            )
        else:
            raise TypeError(
                f"Audio indexing requires int/float or slice, got {type(key).__name__}"
            )

    def _get_segment(self) -> "AudioSegment":
        """Get the AudioSegment for this time range, exact to the frame (see
        ``_frame_bounds``; never padded, never past the last frame)."""
        return self._audio.get_sample_slice(*self._frame_bounds())

    def save(
        self,
        output: Output = None,
        *,
        format: str | None = None,
        bitrate: str = "192k",
        **export_kwargs,
    ) -> Path:
        """
        Save this audio/segment to a new audio file.

        Args:
            output: Where to put the result — None (save beside the input with
                an auto-derived name), a file path, a directory (auto-named), or
                a callable sink. See mixing.egress.
            format: Audio format (mp3, wav, etc.). Auto-detected from extension if None.
            bitrate: Bitrate for compressed formats
            **export_kwargs: Additional arguments for pydub export

        Returns:
            Path to saved file

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> audio[10:30].save("clip.mp3")  # doctest: +SKIP
            >>> audio[10:30].save("clip.wav", format="wav")  # doctest: +SKIP
        """
        # Auto-name (beside the source) when no explicit destination is given.
        if self.src_path:
            src = Path(self.src_path)
            default_name = (
                f"{src.stem}_{int(self.start_time)}_{int(self.end_time)}{src.suffix}"
            )
        else:
            default_name = f"audio_{int(self.start_time)}_{int(self.end_time)}.mp3"

        sink = output if is_sink(output) else None
        if output is None or sink is not None:
            # No path given (or a sink): write to the default location beside
            # the input, then hand that Path to the sink if there is one.
            if self.src_path:
                output_path = Path(self.src_path).with_name(default_name)
            else:
                output_path = Path(default_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = resolve_output_path(output, default_name=default_name)

        # Auto-detect format from extension
        if format is None:
            format = output_path.suffix[1:] if output_path.suffix else "mp3"

        # Export segment
        segment = self._get_segment()
        segment.export(
            str(output_path), format=format, bitrate=bitrate, **export_kwargs
        )

        logger.info("Saved audio to: %s", output_path)
        return sink(output_path) if sink is not None else output_path

    def fade_in(self, duration: float = 1.0) -> "Audio":
        """
        Apply fade-in effect.

        Args:
            duration: Fade duration in seconds

        Returns:
            New Audio with fade applied

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> faded = audio.fade_in(2.0)  # 2 second fade in  # doctest: +SKIP
        """
        segment = self._get_segment()
        duration_ms = int(duration * 1000)
        faded = segment.fade_in(duration_ms)
        return Audio(faded, time_unit=self.time_unit)

    def fade_out(self, duration: float = 1.0) -> "Audio":
        """
        Apply fade-out effect.

        Args:
            duration: Fade duration in seconds

        Returns:
            New Audio with fade applied

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> faded = audio.fade_out(3.0)  # 3 second fade out  # doctest: +SKIP
        """
        segment = self._get_segment()
        duration_ms = int(duration * 1000)
        faded = segment.fade_out(duration_ms)
        return Audio(faded, time_unit=self.time_unit)

    def __add__(self, other: "Audio") -> "Audio":
        """
        Concatenate two audio segments.

        Examples:
            >>> audio1 = Audio("part1.mp3")  # doctest: +SKIP
            >>> audio2 = Audio("part2.mp3")  # doctest: +SKIP
            >>> combined = audio1 + audio2  # doctest: +SKIP
        """
        seg1 = self._get_segment()
        seg2 = other._get_segment()
        combined = seg1 + seg2
        return Audio(combined, time_unit=self.time_unit)

    def overlay(
        self,
        other: "Audio",
        position: float = 0.0,
        *,
        gain_during_overlay: float = 0.0,
    ) -> "Audio":
        """
        Overlay another audio on top of this one.

        Args:
            other: Audio to overlay
            position: Position in seconds where overlay starts
            gain_during_overlay: Volume adjustment in dB during overlay

        Returns:
            New Audio with overlay applied

        Examples:
            >>> bg = Audio("background.mp3")  # doctest: +SKIP
            >>> voice = Audio("voice.mp3")  # doctest: +SKIP
            >>> mixed = bg.overlay(voice, position=5.0, gain_during_overlay=-6)  # doctest: +SKIP
        """
        seg1 = self._get_segment()
        seg2 = other._get_segment()
        position_ms = int(position * 1000)
        mixed = seg1.overlay(
            seg2, position=position_ms, gain_during_overlay=gain_during_overlay
        )
        return Audio(mixed, time_unit=self.time_unit)

    def normalize(self, *, headroom: float = 0.1) -> "Audio":
        """Peak-normalize the audio (via ``pydub.effects.normalize``).

        Boosts (or attenuates) the segment so its loudest peak sits ``headroom``
        dB below 0 dBFS. Pure pydub — adds no new dependency.

        Args:
            headroom: Target peak distance below 0 dBFS, in dB (keyword-only).

        Returns:
            New Audio with normalization applied.

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> louder = audio.normalize()  # doctest: +SKIP
        """
        from pydub import effects

        segment = self._get_segment()
        normalized = effects.normalize(segment, headroom=headroom)
        return Audio(normalized, time_unit=self.time_unit)

    def to_mono(self) -> "Audio":
        """Downmix to a single channel (via pydub ``set_channels(1)``).

        Returns:
            New mono Audio. Pure pydub — adds no new dependency.

        Examples:
            >>> audio = Audio("stereo.mp3")  # doctest: +SKIP
            >>> mono = audio.to_mono()  # doctest: +SKIP
        """
        segment = self._get_segment()
        mono = segment.set_channels(1)
        return Audio(mono, time_unit=self.time_unit)

    def resample(self, sample_rate: int) -> "Audio":
        """Change the sample rate (via pydub ``set_frame_rate``).

        Args:
            sample_rate: Target sample rate in Hz (e.g. ``16000``, ``44100``).

        Returns:
            New Audio at the requested sample rate. Pure pydub — adds no new
            dependency.

        Examples:
            >>> audio = Audio("song.mp3")  # doctest: +SKIP
            >>> downsampled = audio.resample(16000)  # doctest: +SKIP
        """
        segment = self._get_segment()
        resampled = segment.set_frame_rate(sample_rate)
        return Audio(resampled, time_unit=self.time_unit)

    def close(self) -> None:
        """Release the reference to the in-memory audio (no OS handles to free).

        ``Audio`` is fully in-memory (a decoded ``AudioSegment``), so there is
        nothing OS-level to close. ``close`` simply drops the reference so the
        data can be garbage-collected promptly; the object should not be used
        afterwards.
        """
        self._audio = None

    def __enter__(self) -> "Audio":
        """Support ``with Audio(path) as a: ...`` — returns ``self``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release in-memory data on context exit (see :meth:`close`)."""
        self.close()

    def __repr__(self) -> str:
        if self._start_time is not None or self._end_time is not None:
            src_info = f"'{self.src_path}'" if self.src_path else "AudioSegment"
            return (
                f"Audio({src_info}, "
                f"time_unit='{self.time_unit}', "
                f"start={self.start_time:.2f}s, "
                f"end={self.end_time:.2f}s, "
                f"duration={self.duration:.2f}s)"
            )
        else:
            src_info = f"'{self.src_path}'" if self.src_path else "AudioSegment"
            return (
                f"Audio({src_info}, "
                f"time_unit='{self.time_unit}', "
                f"duration={self.full_duration:.2f}s)"
            )

    @property
    def samples(self) -> AudioSamples:
        """Get sample-by-sample Mapping interface for this audio."""
        start_sample = int(self.start_time * self.sample_rate)
        end_sample = int(self.end_time * self.sample_rate)
        return AudioSamples(
            self._audio, start_sample=start_sample, end_sample=end_sample
        )


# Convenience functions


def crop_audio(
    src_path: str,
    start: float | int | None = None,
    end: float | int | None = None,
    *,
    time_unit: AudioTimeUnit = "seconds",
    output: Output = None,
    **save_kwargs,
) -> Path:
    """
    Convenience function to crop and save an audio segment.

    Args:
        src_path: Path to source audio
        start: Start time (None = beginning)
        end: End time (None = end of audio)
        time_unit: Unit for start/end values
        output: Where to put the result — None (save beside the input), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional arguments for save operation

    Returns:
        Path to saved cropped audio

    Examples:
        >>> crop_audio("song.mp3", 10, 30)  # Crop 10s-30s  # doctest: +SKIP
        >>> crop_audio("song.mp3", 44100, 88200, time_unit="samples")  # doctest: +SKIP
    """
    audio = Audio(src_path, time_unit=time_unit)
    segment = audio[start:end]
    return segment.save(output, **save_kwargs)


def fade_in(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-in effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_in("song.mp3", 2.0, output="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_in("song.mp3", 2.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_in(duration)
    return deliver(
        faded,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_fade_in.mp3",
    )


def fade_out(
    src: Union[str, Audio],
    duration: float = 1.0,
    *,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Apply fade-out effect to audio.

    Args:
        src: Audio source (filepath or Audio instance)
        duration: Fade duration in seconds
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> fade_out("song.mp3", 3.0, output="faded.mp3")  # doctest: +SKIP
        >>> audio = fade_out("song.mp3", 3.0)  # Returns Audio instance  # doctest: +SKIP
    """
    audio = Audio(src) if isinstance(src, str) else src
    faded = audio.fade_out(duration)
    return deliver(
        faded,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_fade_out.mp3",
    )


def concatenate_audio(
    *sources: Union[str, Audio],
    output: Output = None,
    crossfade: float = 0.0,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Concatenate multiple audio files/segments.

    Args:
        *sources: Audio sources (filepaths or Audio instances)
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        crossfade: Crossfade duration in seconds between segments
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> concatenate_audio("intro.mp3", "main.mp3", "outro.mp3")  # doctest: +SKIP
        >>> concatenate_audio("a.mp3", "b.mp3", output="combined.mp3")  # doctest: +SKIP
        >>> concatenate_audio("a.mp3", "b.mp3", crossfade=0.5)  # 500ms crossfade  # doctest: +SKIP
    """
    if not sources:
        raise ValueError("At least one audio source is required")

    # Convert all to Audio instances
    audios = [Audio(src) if isinstance(src, str) else src for src in sources]

    # Start with first audio
    result = audios[0]

    # Add remaining audios
    for audio in audios[1:]:
        if crossfade > 0:
            # Apply crossfade
            seg1 = result._get_segment()
            seg2 = audio._get_segment()
            crossfade_ms = int(crossfade * 1000)
            combined = seg1.append(seg2, crossfade=crossfade_ms)
            result = Audio(combined)
        else:
            # Simple concatenation
            result = result + audio

    return deliver(
        result,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_concat.mp3",
    )


#: Gain (dB) below which an overlay/background contribution is treated as muted.
#: -120 dB is ~1e-6 amplitude — inaudible — and avoids ``log10(0) = -inf``.
_MIX_SILENCE_FLOOR_DB = -120.0


def _mix_amplitude_to_db(amplitude: float) -> float:
    """Linear amplitude ratio in ``[0, 1]`` → dB gain, floored at silence."""
    if amplitude <= 0.0:
        return _MIX_SILENCE_FLOOR_DB
    return max(_MIX_SILENCE_FLOOR_DB, float(20.0 * np.log10(amplitude)))


def overlay_audio(
    background: Union[str, Path, Audio],
    overlay: Union[str, Path, Audio],
    position: float = 0.0,
    *,
    mix_ratio: float = 0.5,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """
    Overlay/mix two audio sources.

    ``mix_ratio`` is the prominence of the *overlay*, modeled as a
    linear-amplitude crossfade between background-only and overlay-only: the
    overlay plays at gain ``20·log10(mix_ratio)`` and the background is ducked
    by ``20·log10(1 - mix_ratio)`` for the overlap's duration. So ``0.0`` =
    only the background, ``1.0`` = only the overlay (during the overlap),
    ``0.5`` = an equal blend (both ~-6 dB).

    Args:
        background: Background audio (filepath or Audio instance)
        overlay: Audio to overlay (filepath or Audio instance)
        position: Position in seconds where overlay starts
        mix_ratio: Prominence of the overlay in ``[0.0, 1.0]`` (see above).
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments

    Returns:
        Audio instance or Path to saved file

    Examples:
        >>> overlay_audio("music.mp3", "voice.mp3", position=5.0)  # doctest: +SKIP
        >>> overlay_audio("bg.mp3", "sfx.mp3", mix_ratio=0.3)  # 30% overlay, 70% bg  # doctest: +SKIP
    """
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    bg_audio = (
        Audio(background) if isinstance(background, (str, os.PathLike)) else background
    )
    ov_audio = Audio(overlay) if isinstance(overlay, (str, os.PathLike)) else overlay

    overlay_gain_db = _mix_amplitude_to_db(mix_ratio)
    background_gain_db = _mix_amplitude_to_db(1.0 - mix_ratio)

    if overlay_gain_db <= _MIX_SILENCE_FLOOR_DB:
        # Overlay is muted — the result is just the background, untouched.
        mixed = bg_audio
    else:
        overlay_seg = ov_audio._get_segment()
        if overlay_gain_db != 0.0:
            overlay_seg = overlay_seg + overlay_gain_db  # pydub gain
        overlay_adjusted = Audio(overlay_seg, time_unit=ov_audio.time_unit)
        mixed = bg_audio.overlay(
            overlay_adjusted, position=position, gain_during_overlay=background_gain_db
        )

    return deliver(
        mixed,
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_overlay.mp3",
    )


#: Default crossfade (seconds) applied at every loop join in :func:`loop_audio`.
#: Long enough to hide the waveform discontinuity where the tail meets the head,
#: short enough not to smear the bed's content.
DEFAULT_LOOP_CROSSFADE_S = 0.5

#: A loop crossfade may consume at most this fraction of the source's duration.
#: Each join must still advance the timeline, so this can never reach ``1.0``.
_MAX_LOOP_CROSSFADE_FRACTION = 0.5


def loop_audio(
    source: Union[str, Path, Audio],
    target_duration_s: float,
    *,
    crossfade_s: float = DEFAULT_LOOP_CROSSFADE_S,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Tile an audio source until it fills ``target_duration_s``, seamlessly.

    The source is appended to itself with a ``crossfade_s`` crossfade at every
    join (pydub's equal-gain fade-out/fade-in), then trimmed to *exactly*
    ``target_duration_s``. This is the "make a 20 s ambient bed last 4 minutes"
    primitive: without the crossfade, every loop point is a hard splice and a
    waveform discontinuity you can hear as a click.

    A source **longer** than the target is simply trimmed — looping is only
    ever additive, never a no-op guard the caller has to write.

    Because each join consumes ``crossfade_s`` of timeline, the crossfade is
    clamped to half the source's duration; otherwise a long crossfade over a
    short source would never advance.

    Note the tail is cut wherever ``target_duration_s`` lands (mid-loop is
    normal), and no fade-out is applied — chain :func:`fade_out` if the bed
    ends exposed.

    Args:
        source: Audio to loop (filepath or :class:`Audio`).
        target_duration_s: Duration of the result, in seconds (> 0).
        crossfade_s: Crossfade at each loop join, in seconds. ``0`` gives hard
            splices. Clamped to half the source duration.
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments.

    Returns:
        Audio instance or Path to saved file.

    Examples:
        >>> bed = loop_audio("room_tone.wav", 90.0)  # 90s bed  # doctest: +SKIP
        >>> loop_audio("waves.wav", 240.0, output="bed.wav")  # doctest: +SKIP
    """
    if target_duration_s <= 0:
        raise ValueError(f"target_duration_s must be > 0, got {target_duration_s}")
    if crossfade_s < 0:
        raise ValueError(f"crossfade_s must be >= 0, got {crossfade_s}")

    audio = source if isinstance(source, Audio) else Audio(source)
    segment = audio._get_segment()
    source_ms = len(segment)
    if source_ms <= 0:
        raise ValueError("Cannot loop an empty audio source")

    target_ms = int(round(target_duration_s * 1000))
    crossfade_ms = min(
        int(round(crossfade_s * 1000)),
        int(source_ms * _MAX_LOOP_CROSSFADE_FRACTION),
    )

    looped = segment
    while len(looped) < target_ms:
        looped = looped.append(segment, crossfade=crossfade_ms)
    looped = looped[:target_ms]

    return deliver(
        Audio(looped, time_unit=audio.time_unit),
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_loop.mp3",
    )


#: Gain (dB) applied to the bed while the sidechain is active.
DEFAULT_DUCK_DB = -12.0
#: Sidechain frames louder than this (dBFS RMS) count as "active".
DEFAULT_DUCK_THRESHOLD_DB = -40.0
#: Time constant for moving *down* to the ducked level (sidechain goes active).
DEFAULT_DUCK_ATTACK_S = 0.05
#: Time constant for coming *back up* to unity (sidechain goes quiet).
DEFAULT_DUCK_RELEASE_S = 0.4
#: How long the duck is held past the last active frame, so the bed rides
#: through the gaps between words instead of pumping on every syllable.
DEFAULT_DUCK_HOLD_S = 0.2
#: Level-detector frame size — the time resolution of the sidechain envelope.
DEFAULT_DUCK_FRAME_S = 0.02


def _frame_rms_db(segment: "AudioSegment", *, frame_n: int) -> np.ndarray:
    """Per-frame RMS level (dBFS) of ``segment``, downmixed to mono.

    ``frame_n`` is the frame length in samples-per-channel. The final partial
    frame is zero-padded, which only ever *lowers* its level — a partial frame
    can't fake activity.
    """
    samples = segment_samples(segment).mean(axis=1)

    n_frames = max(1, int(np.ceil(len(samples) / frame_n)))
    padded = np.zeros(n_frames * frame_n, dtype=np.float64)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame_n) ** 2, axis=1))
    floor = 10.0 ** (_MIX_SILENCE_FLOOR_DB / 20.0)
    return 20.0 * np.log10(np.maximum(rms, floor))


def _hold_active(active: np.ndarray, *, hold_frames: int) -> np.ndarray:
    """Extend each active run forward by ``hold_frames`` frames."""
    if hold_frames <= 0:
        return active
    indices = np.arange(len(active))
    last_active = np.maximum.accumulate(np.where(active, indices, -1))
    return (last_active >= 0) & ((indices - last_active) <= hold_frames)


def _duck_gain_envelope(
    active: np.ndarray,
    *,
    frame_s: float,
    duck_db: float,
    attack_s: float,
    release_s: float,
) -> np.ndarray:
    """Per-frame gain (dB) that falls to ``duck_db`` while ``active``, else 0.

    A one-pole smoother in the dB domain with separate attack (going down) and
    release (coming back up) time constants — the standard gain-smoothing shape
    of a broadcast ducker.
    """

    def _coeff(time_constant_s: float) -> float:
        if time_constant_s <= 0:
            return 1.0  # instantaneous
        return 1.0 - float(np.exp(-frame_s / time_constant_s))

    attack_coeff = _coeff(attack_s)
    release_coeff = _coeff(release_s)

    targets = np.where(active, duck_db, 0.0)
    gains = np.empty(len(targets), dtype=np.float64)
    gain = 0.0
    for i, target in enumerate(targets):
        coeff = attack_coeff if target < gain else release_coeff
        gain += (target - gain) * coeff
        gains[i] = gain
    return gains


def _apply_gain_envelope(
    segment: "AudioSegment", gain_db: np.ndarray, *, frame_s: float
) -> "AudioSegment":
    """Apply a per-frame dB gain envelope to ``segment``, sample-interpolated.

    Gains are linearly interpolated between frame centers so the envelope is
    continuous — a per-frame staircase would add its own zipper noise.

    Samples round-trip through numpy at the segment's own width; pydub only
    ever holds 1-, 2- or 4-byte samples (its ``AudioSegment`` widens 24-bit to
    32-bit on construction), so the `array` typecode always matches the frame
    width and no width conversion is needed here.
    """
    AudioSegment = require_package("pydub").AudioSegment

    samples = np.array(segment.get_array_of_samples())
    channels = segment.channels
    per_channel = len(samples) // channels
    if per_channel == 0:
        return segment

    sample_times = (np.arange(per_channel) + 0.5) / segment.frame_rate
    frame_times = (np.arange(len(gain_db)) + 0.5) * frame_s
    gain_linear = 10.0 ** (np.interp(sample_times, frame_times, gain_db) / 20.0)

    scaled = samples.astype(np.float64).reshape(-1, channels) * gain_linear[:, None]
    limits = np.iinfo(samples.dtype)
    adjusted = np.clip(np.rint(scaled), limits.min, limits.max).astype(samples.dtype)

    return AudioSegment(
        data=adjusted.reshape(-1).tobytes(),
        sample_width=segment.sample_width,
        frame_rate=segment.frame_rate,
        channels=channels,
    )


def duck_audio(
    bed: Union[str, Path, Audio],
    sidechain: Union[str, Path, Audio],
    *,
    duck_db: float = DEFAULT_DUCK_DB,
    threshold_db: float = DEFAULT_DUCK_THRESHOLD_DB,
    attack_s: float = DEFAULT_DUCK_ATTACK_S,
    release_s: float = DEFAULT_DUCK_RELEASE_S,
    hold_s: float = DEFAULT_DUCK_HOLD_S,
    frame_s: float = DEFAULT_DUCK_FRAME_S,
    output: Output = None,
    **save_kwargs,
) -> Union[Audio, Path]:
    """Duck ``bed`` wherever ``sidechain`` is loud (sidechain ducking).

    **What it does.** A level-detector sidechain ducker: the ``sidechain``
    (typically the dialogue track) is framed at ``frame_s``, each frame's RMS
    is compared to ``threshold_db``, active frames are extended by ``hold_s``,
    and the resulting on/off signal drives a gain envelope on ``bed`` that
    falls to ``duck_db`` with time constant ``attack_s`` and returns to unity
    with ``release_s``. The envelope is interpolated to sample resolution
    before it is applied, so there is no zipper noise. The result always has
    the **bed's** duration; a shorter sidechain simply leaves the tail
    un-ducked.

    **What it does not do.** It is not a full compressor: there is no ratio,
    knee, or makeup gain — the duck depth is the fixed ``duck_db``, not a
    function of how loud the sidechain is. There is no lookahead, so with a
    short ``attack_s`` the first few milliseconds of a sudden word can sneak
    through at full bed level. Detection is **energy-based, not speech-aware**:
    any loud sidechain content (music, a door slam, hiss above
    ``threshold_db``) ducks the bed just as dialogue would. And the bed is
    attenuated, never EQ'd — it does not carve a vocal-band notch.

    Args:
        bed: The audio to be ducked (filepath or :class:`Audio`).
        sidechain: The audio that triggers ducking — the dialogue track.
        duck_db: Gain (dB, ``<= 0``) held while the sidechain is active.
        threshold_db: Sidechain frames above this RMS dBFS count as active.
        attack_s: Time constant for reaching the ducked level.
        release_s: Time constant for returning to unity.
        hold_s: How long the duck persists after the last active frame.
        frame_s: Level-detector frame size (envelope time resolution).
        output: Where to put the result — None (return the Audio object), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        **save_kwargs: Additional save arguments.

    Returns:
        Audio instance or Path to saved file.

    Examples:
        >>> quiet_bed = duck_audio("bed.wav", "dialogue.wav")  # doctest: +SKIP
        >>> duck_audio("bed.wav", "vo.wav", duck_db=-18)  # deeper duck  # doctest: +SKIP
    """
    if duck_db > 0:
        raise ValueError(f"duck_db must be <= 0 dB (a duck attenuates), got {duck_db}")
    if frame_s <= 0:
        raise ValueError(f"frame_s must be > 0, got {frame_s}")
    for name, value in (
        ("attack_s", attack_s),
        ("release_s", release_s),
        ("hold_s", hold_s),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    bed_audio = bed if isinstance(bed, Audio) else Audio(bed)
    side_audio = sidechain if isinstance(sidechain, Audio) else Audio(sidechain)

    bed_segment = bed_audio._get_segment()
    side_segment = side_audio._get_segment()
    if len(bed_segment) <= 0:
        raise ValueError("Cannot duck an empty bed")

    n_frames = max(1, int(np.ceil((len(bed_segment) / 1000.0) / frame_s)))
    side_frame_n = max(1, int(round(frame_s * side_segment.frame_rate)))

    active = np.zeros(n_frames, dtype=bool)
    if len(side_segment) > 0:
        levels = _frame_rms_db(side_segment, frame_n=side_frame_n)
        overlap = min(n_frames, len(levels))
        active[:overlap] = levels[:overlap] > threshold_db

    gain_db = _duck_gain_envelope(
        _hold_active(active, hold_frames=int(round(hold_s / frame_s))),
        frame_s=frame_s,
        duck_db=duck_db,
        attack_s=attack_s,
        release_s=release_s,
    )
    ducked = _apply_gain_envelope(bed_segment, gain_db, frame_s=frame_s)

    return deliver(
        Audio(ducked, time_unit=bed_audio.time_unit),
        output,
        write=lambda a, p: a.save(p, **save_kwargs),
        default_name="audio_ducked.mp3",
    )


def save_audio_clip(
    audio_src: str | None = None,
    start: float = 0,
    end: float | None = None,
    *,
    time_unit: AudioTimeUnit | None = None,
    output: Output = None,
    format: str = "mp3",
) -> Path:
    """
    Extract and save an audio clip.

    Args:
        audio_src: Path to audio file. If None, gets from clipboard.
        start: Start time/sample (default: 0)
        end: End time/sample (None = end of audio)
        time_unit: Unit for start/end ('seconds', 'samples', 'milliseconds')
        output: Where to put the result — None (save beside the input), a file
            path, a directory (auto-named), or a callable sink. See mixing.egress.
        format: Output format

    Returns:
        Path to saved audio file

    Examples:
        >>> save_audio_clip("song.mp3", 10, 30)  # Save 10s-30s  # doctest: +SKIP
        >>> save_audio_clip(start=5, end=15)  # From clipboard  # doctest: +SKIP
    """
    if audio_src is None:
        audio_src = get_path_from_clipboard()

    if time_unit is None:
        time_unit = "seconds"

    audio = Audio(audio_src, time_unit=time_unit)
    segment = audio[start:end] if end is not None else audio[start:]

    return segment.save(output, format=format)


def find_audio_offset(
    reference_audio: AudioSource,
    query_audio: AudioSource,
    *,
    sample_rate: int = 16000,
) -> float:
    """Find the time offset where query_audio best aligns within reference_audio.

    Uses FFT-based cross-correlation to find the position in reference_audio
    where query_audio starts. This is useful for aligning different recordings
    of the same performance — for example, aligning a studio recording (voice
    + instruments) with a camera recording (voice only).

    The two audio signals don't need to be identical; they just need to share
    a correlated component (e.g., the same voice in both).

    Args:
        reference_audio: The longer audio to search within (e.g., extracted
            from a video). Accepts a file path, numpy array, or AudioSegment.
        query_audio: The shorter audio to align (e.g., a studio recording).
            Accepts a file path, numpy array, or AudioSegment.
        sample_rate: Sample rate for analysis. Lower values are faster but
            less precise. Default 16000 Hz gives ~0.06ms precision, which is
            more than sufficient for alignment purposes.

    Returns:
        Offset in seconds — the position in reference_audio where query_audio
        starts. Positive means query begins after the start of reference.

    Examples:
        >>> from mixing.audio import find_audio_offset  # doctest: +SKIP
        >>> # Find where a studio recording aligns with a camera recording
        >>> offset = find_audio_offset("camera_audio.wav", "studio.mp3")  # doctest: +SKIP
        >>> print(f"Studio recording starts at {offset:.2f}s in the camera audio")  # doctest: +SKIP
    """
    return find_audio_offset_detailed(
        reference_audio, query_audio, sample_rate=sample_rate
    ).offset_s


@dataclass(frozen=True)
class AudioOffset:
    """The result of aligning one recording within another.

    Attributes:
        offset_s: Time in ``reference`` where ``query`` begins (seconds). Positive
            means query starts after the reference's t=0; **negative** means query
            began before it (e.g. a phone that started filming before the song).
        confidence: A scale-invariant normalized cross-correlation coefficient in
            ``[0, 1]`` at the best lag — comparable ACROSS clips of different loudness
            and length. ~0.5+ is a strong match; near 0 means no shared component.
        sample_rate: The analysis sample rate the offset was computed at.
    """

    offset_s: float
    confidence: float
    sample_rate: int


def _load_mono_samples(source: AudioSource, sample_rate: int) -> np.ndarray:
    """Load any :data:`AudioSource` as a mono float64 array at ``sample_rate``.

    Decodes at the source's NATIVE rate (pydub → ffmpeg), downmixes to mono,
    then resamples with an anti-aliased polyphase filter
    (:func:`mixing.audio.samples.resample`, which low-pass filters before
    decimating). pydub's ``set_frame_rate`` is deliberately NOT
    used for the rate change: it delegates to ``audioop.ratecv`` — linear
    interpolation with no anti-alias filter — so downsampling a 48 kHz camera
    track to the 16 kHz analysis rate folds everything above 8 kHz back over
    the band. Measured on real multicam footage that aliasing roughly HALVED
    the alignment confidence for identical (correct) offsets, making the score
    depend on the decode path rather than the signal (issue #25).
    """
    seg = _normalize_audio_source(source, target_type="AudioSegment")
    seg = seg.set_channels(1)
    # raw integer values, deliberately unscaled: the alignment features and
    # their tests (test_native_rate_load_is_lossless) work on these levels
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    return resample(samples, seg.frame_rate, sample_rate)


def _xcorr_surface(
    ref: np.ndarray, query: np.ndarray, *, min_overlap_ratio: float = 0.5
) -> "tuple[np.ndarray, np.ndarray]":
    """``(lags, scores)`` — the whole overlap-normalized correlation surface.

    ``scores`` is ``abs(coefficient)`` in ``[0, 1]`` at every admissible lag and ``-1.0``
    at the lags excluded by ``min_overlap_ratio``, so an argmax over it is already the
    guarded one. Split out of :func:`_normalized_xcorr` because the SHAPE of this surface
    is the evidence a repetitive reference destroys: its argmax alone cannot say which
    repeat a window came from, and only a caller that can see the rival peaks
    (:func:`_candidate_lags`) has anything to put to a vote.

    **Normalize per lag, by the energy of that lag's overlap** — the trap named in issue
    #30. An unnormalized FFT correlation lets a boundary spike in the reference make
    every window lock onto the same lag, which reads downstream as a confident and
    entirely fictitious "everything aligns at t≈0".
    """
    correlate = require_package("scipy.signal").correlate
    ref = ref - ref.mean()
    query = query - query.mean()
    n_r, n_q = len(ref), len(query)
    num = correlate(ref, query, mode="full", method="fft")
    # lag L for full-output index k is L = k - (n_q - 1)
    lags = np.arange(n_r + n_q - 1) - (n_q - 1)
    overlap = np.minimum(n_r, lags + n_q) - np.maximum(0, lags)
    # windowed sum-of-squares of each signal over its overlap region, via cumsum
    cum_r = np.concatenate([[0.0], np.cumsum(ref**2)])
    cum_q = np.concatenate([[0.0], np.cumsum(query**2)])
    r_lo = np.maximum(0, lags)
    q_lo = np.maximum(0, -lags)
    energy_r = cum_r[r_lo + overlap] - cum_r[r_lo]
    energy_q = cum_q[q_lo + overlap] - cum_q[q_lo]
    denom = np.sqrt(energy_r * energy_q)
    # Divide only where the denominator is non-zero. `np.where(denom > 0, num / denom, 0)`
    # evaluates the quotient for EVERY lag first, so a silent overlap (denom == 0, which a
    # real recording reaches at its extreme lags) raised a divide-by-zero RuntimeWarning on
    # every call before the mask was applied.
    coeff = np.zeros_like(denom)
    nz = denom > 0
    np.divide(num, denom, out=coeff, where=nz)
    valid = overlap >= max(1.0, min_overlap_ratio * min(n_r, n_q))
    return lags, np.where(valid, np.abs(coeff), -1.0)


def _normalized_xcorr(
    ref: np.ndarray, query: np.ndarray, *, min_overlap_ratio: float = 0.5
) -> tuple[int, float]:
    """Overlap-normalized cross-correlation of two mono signals.

    Returns ``(lag_samples, coefficient)`` where ``lag_samples`` is where ``query``
    begins within ``ref`` (may be negative) and ``coefficient`` is the normalized
    cross-correlation in ``[0, 1]`` at that lag. Normalizing each lag by the energy of
    its actual overlap makes the score scale-invariant (comparable across clips) AND
    removes the triangular-overlap argmax bias for clips that extend before/after the
    reference — the common multi-device case. Lags overlapping less than
    ``min_overlap_ratio`` of the shorter signal are excluded so a tiny sliver of overlap
    can't win.

    **The argmax is only trustworthy when the surface has one peak.** On material that
    repeats verbatim it does not, and the rival peaks are near-tied — so this returns a
    near-coin-flip with a coefficient that looks excellent (issue #30). Callers that can
    be handed repetitive material go through :func:`_candidate_lags` instead.
    """
    lags, scored = _xcorr_surface(ref, query, min_overlap_ratio=min_overlap_ratio)
    best = int(np.argmax(scored))
    return int(lags[best]), float(max(scored[best], 0.0))


#: How close a rival correlation peak must score to the best one before it counts as a
#: NEAR-TIE — a fraction of the best score. Measured on real music (issue #30), the
#: second peak reaches 0.987-0.993 of the first at musical periods, and on synthetic
#: repeats the two are within 0.2%. At that separation the argmax is a coin flip, so
#: everything inside this band is treated as "the correlation has no opinion" and the
#: choice is handed to the vote. Above it, a peak wins on its own evidence — which is
#: what keeps a genuine stop/restart free to depart from its neighbours.
NEAR_TIE_RATIO = 0.05

#: How many independent windows it takes before "how many of them agree" is a fact
#: rather than a tautology. One window agrees with itself, so a support fraction
#: computed over a single vote is always 1.0 and has measured nothing — and a
#: manufactured 1.0 is worse than no number at all, because it VOUCHES. Below this,
#: support is reported as ``None``.
MIN_WINDOWS_FOR_SUPPORT = 2

#: How much of its neighbour a window may SHARE and still be counted as a second opinion
#: when support is tallied. Two windows overlapping 95% of their samples are one opinion
#: read twice: they see the same content, so they inherit the same bias and agree for no
#: reason worth reporting. Measured on real material at ``window_s=9.5, hop_s=0.5``, that
#: is exactly how an offset 102 s from the truth came back at ``support=1.00`` — the same
#: shared-bias failure as issue #30, one level up. Windows closer together than this are
#: still measured, still vote, and still set the offset; they are only excluded from the
#: TALLY, whose whole meaning is "how many independent looks agree".
#:
#: The default hop (:data:`SPAN_HOP_S`, half a window) sits exactly ON this bound, so every
#: window of the regular grid counts. The one window that does not is the TAIL window
#: :func:`_window_offsets` appends when a clip's length is not a whole number of hops: it
#: starts wherever it must to reach the end, usually less than half a window after its
#: neighbour, so it is excluded and the denominator loses one. Measured on real material,
#: that moved one clip's support from 0.45 to 0.50 — the offsets did not move. It is the
#: right call and it does cost something: the tail window is the only look at the clip's
#: last stretch, and heavily-overlapped agreement is exactly what must not be counted, so
#: the choice is between a look nothing corroborates and a corroboration that is an echo.
MAX_SUPPORT_OVERLAP = 0.5

#: What one window's BALLOT MENTION is worth in the support tally, against the 1.0 of an
#: independent argmax. A window that named the winning offset outright found it unaided;
#: a window that merely could not separate it from its own answer
#: (:data:`NEAR_TIE_RATIO`) is weaker evidence than that and stronger than nothing, so it
#: casts half a vote rather than none (issue #45). The mention is then scaled by the
#: candidate's share of that window's own best score — both :func:`_dual_confidence`
#: values, so the ratio is a WITHIN-WINDOW ranking and not a calibrated coefficient.
#:
#: Why half rather than some other fraction: at ``0.5`` the number reads as a scale with
#: a boundary a caller can use. Ballot mentions alone can never carry the tally past
#: ``0.5``, so **``support > 0.5`` means at least one independent window reached this
#: offset on its own** — the old statistic's whole question, still answerable, now as the
#: top half of a range instead of the whole of it.
#:
#: What it fixes: an argmax is a real opinion at a 20 s window and close to a coin flip
#: at 4 s, so a bare argmax tally got *less* confident exactly as the fitted window
#: (issue #41) made the estimator *more* reliable. Measured on real cross-device material
#: (issue #45), 21 alignments that were all correct reported 0.00-1.00 with six at 0.00 —
#: the vote landed on the right offset while no individual window's argmax agreed. A gate
#: at 0.25 refused about half of them.
BALLOT_VOTE_WEIGHT = 0.5

#: Most near-tied lags one window may put forward. A cap, not a target: an exactly
#: tiling reference offers one candidate per repeat, and the vote does not get better
#: for counting all of them.
MAX_CANDIDATE_LAGS = 8


def _candidate_lags(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    min_overlap_ratio: float,
    near_tie_ratio: float,
    min_separation: int,
    max_candidates: int = MAX_CANDIDATE_LAGS,
) -> "list[tuple[int, float]]":
    """The near-tied lags this correlation genuinely cannot choose between.

    Returns ``[(lag_samples, coefficient), ...]`` ordered best-first and always
    non-empty, so ``result[0]`` is exactly what :func:`_normalized_xcorr` would have
    returned. A ``near_tie_ratio`` of ``0.0`` therefore reduces this to the plain argmax.

    Only LOCAL MAXIMA are eligible, and each accepted peak suppresses everything within
    ``min_separation`` samples of it — otherwise the shoulders of one peak would flood
    the ballot and a single lag would out-vote every real rival.
    """
    lags, scored = _xcorr_surface(ref, query, min_overlap_ratio=min_overlap_ratio)
    top = int(np.argmax(scored))
    best = float(scored[top])
    if best <= 0 or near_tie_ratio <= 0:
        return [(int(lags[top]), max(best, 0.0))]
    interior = scored[1:-1]
    peaks = np.flatnonzero((interior >= scored[:-2]) & (interior > scored[2:])) + 1
    # The argmax may sit on an edge, where "local maximum" is undefined; it is always a
    # candidate, and always the first one.
    eligible = np.unique(np.concatenate([[top], peaks]))
    eligible = eligible[scored[eligible] >= best * (1.0 - near_tie_ratio)]
    order = eligible[np.argsort(-scored[eligible], kind="stable")]
    picked: list[int] = []
    for k in map(int, order):
        if all(abs(k - j) >= min_separation for j in picked):
            picked.append(k)
            if len(picked) >= max_candidates:
                break
    return [(int(lags[k]), float(scored[k])) for k in picked]


#: Envelope hop, in samples at the analysis rate. 160 @ 16 kHz = 10 ms frames (100 Hz).
ENVELOPE_HOP = 160
#: STFT window for the envelope. 1024 @ 16 kHz = 64 ms — long enough to resolve a musical
#: onset, short enough not to smear it.
ENVELOPE_NFFT = 1024


def onset_envelope(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop: int = ENVELOPE_HOP,
    nfft: int = ENVELOPE_NFFT,
) -> "tuple[np.ndarray, float]":
    """A channel-robust onset/energy envelope: ``(envelope, envelope_rate_hz)``.

    Log-compressed STFT magnitude, positive first difference, summed over frequency, then
    standardized. This is *spectral flux* — it tracks WHEN energy arrives, not the waveform
    itself, so it survives the things that destroy raw-waveform similarity between two
    devices recording the same sound: different microphone responses, different positions
    (hence different room impulse responses), and the resulting phase differences.

    Why this matters, measured on a real 6-device shoot: raw-waveform correlation scored
    provably-correct alignments (three independent methods agreeing to within 10 ms) at
    0.064-0.148, while the envelope scored the same pairs at 0.441-0.634 and a genuine
    non-match at 0.102. The raw coefficient could not separate match from non-match; the
    envelope separates them by more than 4x.

    Args:
        samples: Mono samples.
        sample_rate: Their rate.
        hop: Frames advance by this many samples (sets the envelope's time resolution).
        nfft: STFT window length.

    Returns:
        ``(envelope, envelope_rate_hz)``. The envelope is zero-mean, unit-variance, so
        correlations of two envelopes are directly comparable.
    """
    stft = require_package("scipy.signal").stft
    _, _, spec = stft(
        samples,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=nfft - hop,
        padded=False,
        boundary=None,
    )
    logmag = np.log1p(np.abs(spec))
    flux = np.diff(logmag, axis=1)
    flux[flux < 0] = 0.0  # onsets only — decays carry no timing information
    env = flux.sum(axis=0)
    return (env - env.mean()) / (env.std() + 1e-9), sample_rate / hop


def _envelope_then_waveform(
    ref: np.ndarray,
    query: np.ndarray,
    sample_rate: int,
    *,
    min_overlap_ratio: float,
    ref_envelope: "tuple[np.ndarray, float] | None" = None,
) -> "tuple[int, float]":
    """Waveform picks the lag; the confidence is the better of two views of that lag.

    Returns ``(lag_samples, confidence)``.

    **The waveform locates, and here it locates alone.** Its peak *position* is reliable —
    on a real 6-device shoot it agreed with two independent methods to within 10 ms — and
    it has full sample resolution. It does not get to choose alone how good the match is,
    and the envelope does not get to choose the lag *by itself*: on signals whose energy is
    smoothly modulated rather than percussive (linear chirps under a 1.7 Hz AM), the
    envelope autocorrelation is periodic, and letting it pick moved a known -2.0 s offset
    to -2.5 s.

    **This is the single-shot path, and it is a weaker estimator than the windowed one.**
    Where the envelope's answer can be put to a vote across windows,
    :func:`_feature_candidates` lets the envelope nominate too and the evidence decides —
    which is the only way a cross-device offset the waveform cannot see is ever reached
    (issue #30). One correlation over a whole clip has no vote to hold, so it keeps the
    waveform's location and reports the better of the two scores for it. A caller on
    cross-device material wants :func:`align_clips_to_reference`, whose consensus path is
    on by default.

    **The confidence is the larger of two correlations evaluated AT that lag** — the
    waveform's and the onset envelope's. Not a trick to inflate the score: each feature is
    *blind* in a regime the other sees clearly, and which regime you are in is not knowable
    in advance.

    - *Same source* (an export against its master; a clip cut from the reference): the
      waveform correlates near 1.0. The envelope may be near-useless here, because content
      with no percussive onsets has no flux structure to correlate — measured at 0.08 for an
      exact copy of an onset-free signal.
    - *Different devices* (the multicam case): the waveform is near-useless, because two
      microphones in a room are not sample-correlated even when the alignment is exact —
      measured at 0.055-0.188 for alignments confirmed correct by two other methods. The
      envelope reads the same pairs at 0.17-0.60.
    - *Unrelated audio*: both are low, so the maximum is low and the pair is still rejected —
      measured at max(0.010, 0.021) on a genuine non-match.

    Taking the maximum means the score answers "how similar are these at the offset we are
    reporting, by the most favourable of two complementary measures". Taking either alone
    means silently returning a near-zero score for a perfectly good alignment in the other
    feature's blind spot — which, at a downstream threshold, deletes the user's footage.
    """
    # `ref_envelope` lets a caller that aligns MANY queries against ONE reference
    # compute the reference's envelope once. Omitting it reproduces the previous
    # behaviour exactly, so `find_audio_offset_detailed` is unchanged; supplying it
    # measured 2.11x on `aligned_spans`, which was recomputing it per window.
    env_ref, env_rate = (
        onset_envelope(ref, sample_rate) if ref_envelope is None else ref_envelope
    )
    env_query, _ = onset_envelope(query, sample_rate)
    wav_lag, _ = _normalized_xcorr(ref, query, min_overlap_ratio=min_overlap_ratio)
    return wav_lag, _dual_confidence(
        ref,
        query,
        wav_lag,
        env_ref=env_ref,
        env_query=env_query,
        lag_to_env=env_rate / sample_rate,
    )


def _correlation_at_lag(ref: np.ndarray, query: np.ndarray, lag: int) -> float:
    """Normalized correlation of ``query`` against ``ref`` at exactly ``lag``, in ``[0, 1]``.

    The single-lag counterpart of :func:`_normalized_xcorr`, which searches for the best
    lag. Used to score a lag that some other feature chose.
    """
    n_r, n_q = len(ref), len(query)
    r_lo, q_lo = max(0, lag), max(0, -lag)
    overlap = min(n_r, lag + n_q) - r_lo
    if overlap <= 0:
        return 0.0
    a = ref[r_lo : r_lo + overlap]
    b = query[q_lo : q_lo + overlap]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0:
        return 0.0
    return float(abs(np.dot(a, b)) / denom)


#: How far a waveform refinement may move a lag the envelope located, in envelope hops.
#: One hop IS the envelope's time resolution, so one hop is the whole uncertainty the
#: refinement exists to remove — it buys back sample precision without letting the
#: waveform relocate the window, which is the defect of issue #30.
ENVELOPE_REFINE_HOPS = 1


def _refine_lag_on_waveform(
    ref: np.ndarray,
    query: np.ndarray,
    lag: int,
    *,
    radius: int,
    min_overlap_ratio: float,
) -> int:
    """The best waveform lag within ``radius`` samples of ``lag``.

    A lag located on the onset envelope is quantized to the envelope's hop — 10 ms at the
    default settings, which is visible as lip-sync error. The waveform has full sample
    resolution, so it is asked WHERE INSIDE that hop the lag falls, and nothing else: the
    search sees a slice of the reference only ``radius`` samples wider than the query on
    each side, so it can sharpen the envelope's answer but never move it to a different
    peak. That bound is the point — an unbounded waveform search is issue #30 itself.

    Inside the radius the waveform can still be wrong: a periodic bed (a looped backing
    track, a sustained tone) has a correlation cycle shorter than one envelope hop, so
    this may lock onto the neighbouring cycle rather than the true one. That error is
    bounded by ``radius`` — at most one envelope hop, 10 ms at the defaults — which is
    the resolution the envelope had to offer in the first place, so the refinement never
    leaves the answer worse than the lag it was handed.
    """
    lo = max(0, lag - radius)
    hi = min(len(ref), lag + len(query) + radius)
    if hi - lo <= 0:
        return lag
    lags, scored = _xcorr_surface(
        ref[lo:hi], query, min_overlap_ratio=min_overlap_ratio
    )
    near = np.abs(lags - (lag - lo)) <= radius
    if not near.any():
        return lag
    return int(lags[int(np.argmax(np.where(near, scored, -1.0)))]) + lo


def _dual_confidence(
    ref: np.ndarray,
    query: np.ndarray,
    lag: int,
    *,
    env_ref: np.ndarray,
    env_query: np.ndarray,
    lag_to_env: float,
) -> float:
    """The larger of the waveform's and the envelope's correlation AT ``lag``.

    The scoring half of :func:`_envelope_then_waveform`, factored out so that a lag
    nominated by either feature is scored the same way — a comparison between two
    candidates means nothing if one is scored by a measure the other never faced.

    Each feature is blind in a regime the other sees clearly: an export against its own
    master correlates near 1.0 in the waveform and can have no onset structure at all,
    while two microphones in a room are not sample-correlated even when the alignment is
    exact. The maximum answers "how similar are these here, by the most favourable of two
    complementary views", and an unrelated pair is still low in both.

    **What makes the maximum safe is that the two regimes are DISJOINT, not that the two
    numbers are calibrated against each other.** A waveform coefficient and an envelope
    coefficient are not the same quantity and comparing them as if they were would be
    meaningless. What is true is that where one feature reads high the other reads low —
    same-source material puts the waveform near 1.0 and the envelope near noise,
    cross-device material does the reverse — so the maximum is in practice "the reading
    from whichever feature can see", not a contest between two comparable scales. On
    material that broke that separation the maximum would be picking between numbers that
    do not mean the same thing, and neither this function nor its callers would notice.
    """
    at_waveform = _correlation_at_lag(ref, query, lag)
    if env_ref.size < 2 or env_query.size < 2:  # too short to have an envelope
        return at_waveform
    at_envelope = _correlation_at_lag(env_ref, env_query, int(round(lag * lag_to_env)))
    return max(at_waveform, at_envelope)


def _best_separated(
    nominations: "Sequence[Sequence[tuple[int, float]]]",
    *,
    min_separation: int,
    max_candidates: int = MAX_CANDIDATE_LAGS,
) -> "list[tuple[int, float]]":
    """One ballot from several nominating domains — best-first, one slot each guaranteed.

    ``nominations`` is one ``[(lag, confidence), ...]`` list PER DOMAIN. The result is
    every surviving lag ordered by confidence, with two rules:

    - **No lag twice.** A lag within ``min_separation`` of an already-accepted one is the
      same peak found again; keeping both would let it out-vote a genuine rival for no
      reason other than having been nominated twice.
    - **Every domain that nominated is on the ballot.** Each domain's best surviving lag
      is admitted before ``max_candidates`` may be spent, and survives the cap. Without
      that reservation the cap is a silent failure mode rather than a budget: a reference
      that repeats verbatim gives the waveform ``MAX_CANDIDATE_LAGS`` near-tied aliases,
      every one of them scoring above a cross-device envelope match, so the envelope's
      only nominee is evicted and the ballot is waveform-only again — issue #30 exactly,
      restored by an off-by-a-budget. A domain whose best lag is suppressed as a
      duplicate is already represented and does not get a substitute slot.
    """
    picked: "list[tuple[int, float]]" = []

    def admit(candidate: "tuple[int, float]") -> None:
        lag = candidate[0]
        if all(abs(lag - other) >= min_separation for other, _ in picked):
            picked.append(candidate)

    ranked = [
        sorted(domain, key=lambda c: -c[1]) for domain in nominations if len(domain) > 0
    ]
    # Strongest domain first, so that when two domains nominate the same peak the slot is
    # spent on the one that scored it higher.
    for domain in sorted(ranked, key=lambda d: -d[0][1]):
        admit(domain[0])
    for candidate in sorted(
        (c for domain in ranked for c in domain), key=lambda c: -c[1]
    ):
        if len(picked) >= max_candidates:
            break
        admit(candidate)
    return sorted(picked, key=lambda c: -c[1])


def _feature_candidates(
    ref: np.ndarray,
    query: np.ndarray,
    sample_rate: int,
    *,
    feature: str,
    min_overlap_ratio: float,
    near_tie_ratio: float,
    min_separation: int,
    ref_envelope: "tuple[np.ndarray, float] | None" = None,
) -> "list[tuple[int, float]]":
    """The near-tied lags this window cannot choose between — best-first, by score.

    ``'waveform'`` is :func:`_candidate_lags` on the raw signals and nothing else.
    ``'envelope'`` **also correlates the two onset envelopes and nominates the near-tied
    peaks of that surface**, refined to sample precision on the waveform
    (:func:`_refine_lag_on_waveform`); every nomination, from either feature, is then
    scored by :func:`_dual_confidence` and the list is ordered by that score.

    **Nomination is where the feature choice has to bite.** An earlier version generated
    candidates with :func:`_candidate_lags` on the raw waveform alone and used the
    envelope only to RE-SCORE them. Measured on real cross-device material (issue #30),
    that made ``feature='envelope'`` and ``feature='waveform'`` return byte-identical
    offsets — the flag moved the confidence and never the location — and it put an offset
    15 s from the truth on every window's ballot while the true one reached none of them.
    No vote can select an answer nobody nominated: :func:`_consensus_choice` cancels
    errors that DIFFER across windows, and a feature-domain bias is shared by every
    window, so consensus ratifies it instead and hands back ``support=1.0`` for it.

    **Both features nominate; the evidence chooses.** Which feature can see is not
    knowable in advance and is not a property of the caller's intent — it is a property
    of this pair of signals — so ``'envelope'`` does not mean "ignore the waveform", it
    means "the envelope also gets to put its answer on the ballot". Material whose
    envelope carries no timing information (a slow chirp under a steady tremolo: its
    spectral flux is the tremolo, identical everywhere) nominates noise, and that
    nomination loses on score to a waveform peak correlating near 1.0. Material recorded
    on two devices is the mirror image: the waveform's peak is junk with a junk score,
    and the envelope's is the one that carries evidence.
    """
    waveform_candidates = _candidate_lags(
        ref,
        query,
        min_overlap_ratio=min_overlap_ratio,
        near_tie_ratio=near_tie_ratio,
        min_separation=min_separation,
    )
    if feature != "envelope":
        return waveform_candidates
    env_ref, env_rate = (
        onset_envelope(ref, sample_rate) if ref_envelope is None else ref_envelope
    )
    env_query, _ = onset_envelope(query, sample_rate)
    if env_ref.size < 2 or env_query.size < 2:  # too short to have an envelope
        return waveform_candidates
    hop = sample_rate / env_rate
    # `_candidate_lags` normalizes every lag by the energy of its own overlap
    # (:func:`_xcorr_surface`), and in the envelope domain that is load-bearing rather
    # than cosmetic: an envelope opens on the onset spike of the reference's first frame,
    # and unnormalized, windows lock onto it and report that everything starts at t=0
    # (measured on real material, issue #30).
    envelope_candidates = _candidate_lags(
        env_ref,
        env_query,
        min_overlap_ratio=min_overlap_ratio,
        near_tie_ratio=near_tie_ratio,
        min_separation=max(1, int(round(min_separation / hop))),
    )
    radius = max(1, int(round(ENVELOPE_REFINE_HOPS * hop)))
    refined = [
        _refine_lag_on_waveform(
            ref,
            query,
            int(round(env_lag * hop)),
            radius=radius,
            min_overlap_ratio=min_overlap_ratio,
        )
        for env_lag, _ in envelope_candidates
    ]

    def score(lags: "list[int]") -> "list[tuple[int, float]]":
        return [
            (
                lag,
                _dual_confidence(
                    ref,
                    query,
                    lag,
                    env_ref=env_ref,
                    env_query=env_query,
                    lag_to_env=env_rate / sample_rate,
                ),
            )
            for lag in lags
        ]

    # Kept as two lists, not concatenated: `_best_separated` guarantees each domain a
    # slot, and a flat list would let the waveform's near-tied aliases on a repeating
    # reference spend the whole ballot and evict the envelope's only nominee.
    return _best_separated(
        [score([lag for lag, _ in waveform_candidates]), score(refined)],
        min_separation=min_separation,
    )


#: Alignment features. ``'waveform'`` is raw normalized cross-correlation — correct when
#: both signals come from the SAME source (re-aligning an export against its master) and
#: misleading across devices. ``'envelope'`` adds the onset envelope: it scores every lag
#: by the better of the two views (:func:`_dual_confidence`), and under ``consensus`` it
#: also lets the envelope NOMINATE lags the waveform would never have offered
#: (:func:`_feature_candidates`), which is what makes the choice move the offset and not
#: only the confidence (issue #30). Whole-clip callers without consensus
#: (:func:`find_audio_offset_detailed`) still locate on the waveform alone — see
#: :func:`_envelope_then_waveform` for the material that justifies it.
ALIGNMENT_FEATURES = ("envelope", "waveform")


def find_audio_offset_detailed(
    reference_audio: AudioSource,
    query_audio: AudioSource,
    *,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
    feature: str = "waveform",
) -> AudioOffset:
    """Align ``query_audio`` within ``reference_audio`` — offset **and** confidence.

    The detailed twin of :func:`find_audio_offset` (which returns just ``offset_s``).
    Uses an overlap-normalized cross-correlation so the confidence is a scale-invariant
    coefficient in ``[0, 1]`` — usable both as a per-clip trust gate and to compare
    alignments across clips (which the multi-device / multicam case needs). Unlike the
    scalar helper's assumption that ``reference`` is the longer signal, this handles a
    ``query`` that is longer than, or starts before, the reference (negative offset).

    Args:
        reference_audio: The signal to align within (e.g. the clean song).
        query_audio: The signal to locate (e.g. a phone recording of the song).
        sample_rate: Analysis sample rate (mono). 16 kHz gives ~0.06 ms precision.
        min_overlap_ratio: Reject lags overlapping less than this fraction of the
            shorter signal (guards against a tiny-overlap spurious peak).
        feature: Which similarity feature the confidence is measured on — see
            :data:`ALIGNMENT_FEATURES`. **Here it moves only the confidence**: one
            correlation over a whole clip has no vote to hold, so the lag is the
            waveform's either way (:func:`_envelope_then_waveform`). The windowed
            :func:`align_clips_to_reference` is where the choice also moves the offset.
            Defaults to ``'waveform'`` here because this is the
            low-level primitive and the caller knows their own signals; use ``'envelope'``
            whenever the two recordings came from **different devices**, where a waveform
            coefficient understates a correct alignment several-fold.
            :func:`align_clips_to_reference` — the multi-device primitive — defaults to
            ``'envelope'`` for that reason.

    Returns:
        An :class:`AudioOffset` (``offset_s``, ``confidence``, ``sample_rate``).
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    ref = _load_mono_samples(reference_audio, sample_rate)
    query = _load_mono_samples(query_audio, sample_rate)
    if feature == "envelope":
        lag, coeff = _envelope_then_waveform(
            ref, query, sample_rate, min_overlap_ratio=min_overlap_ratio
        )
    else:
        lag, coeff = _normalized_xcorr(ref, query, min_overlap_ratio=min_overlap_ratio)
    return AudioOffset(
        offset_s=lag / sample_rate, confidence=coeff, sample_rate=sample_rate
    )


#: Default analysis window for :func:`aligned_spans`, in seconds. Long enough that a
#: window of a real recording carries enough structure to correlate, short enough that a
#: span boundary is locatable. This is the knob that trades boundary resolution against
#: cost: halving it doubles the number of correlations.
SPAN_WINDOW_S = 20.0

#: Default hop between windows. Half the window, so every instant of the clip is covered
#: by two windows and a boundary can never fall in the blind spot between them.
SPAN_HOP_S = 10.0

#: How many window LENGTHS the adaptive rule tries to fit into a clip whose caller did not
#: choose a window (:func:`_clip_window_and_hop`). One more than
#: :data:`MIN_WINDOWS_FOR_SUPPORT`, and the margin is the point: aiming AT the quorum would
#: leave a clip exactly on it, so the one window the tally excludes — the tail
#: :func:`_window_offsets` appends when a clip is not a whole number of hops
#: (:data:`MAX_SUPPORT_OVERLAP`) — would drop it back under and support would read ``None``
#: again. Three window lengths at a half-window hop is five windows, of which at least
#: three are independent looks.
MIN_WINDOWS_FOR_SUPPORT_TARGET = MIN_WINDOWS_FOR_SUPPORT + 1

#: The shortest window the adaptive rule will choose, counted in ONSET-ENVELOPE FRAMES
#: rather than seconds: under the default ``feature='envelope'`` it is the envelope that
#: nominates the lag, so :data:`ENVELOPE_HOP` is what a window's real resolution is made
#: of, and the floor has to follow the analysis rate rather than assume one. 300 frames is
#: 3.0 s at 16 kHz — the shortest window measured to land all three clips of a real
#: cross-device shoot on their true offsets (issue #41; 3, 4, 5 and 6 s windows all did,
#: the default 20 s did not). Below this a window stops carrying enough onsets to
#: correlate, and shrinking further buys votes by making each one worthless.
ADAPTIVE_WINDOW_MIN_FRAMES = 300

#: The hop the adaptive rule pairs with the window it picks, as a fraction of that window.
#: It is the DEFAULT pair's own ratio rather than a new number, so an adapted grid has the
#: same shape as the default one: every instant covered by two windows, and every window of
#: the regular grid exactly on the :data:`MAX_SUPPORT_OVERLAP` bound, hence counted.
ADAPTIVE_HOP_RATIO = SPAN_HOP_S / SPAN_WINDOW_S

#: A window must reach this for its span to exist at all.
SPAN_MIN_CONFIDENCE = 0.15

#: How far two windows' implied offsets may disagree and still be called the same span.
#: Wider than a frame at any sane rate (a span is not a cut point) but far tighter than
#: the drift a genuine stop/restart introduces.
SPAN_OFFSET_TOLERANCE_S = 0.25

#: How long an UNVERIFIED gap between two same-offset spans may be and still be treated
#: as one take. Defaults to one window.
#:
#: A dozen seconds of silence mid-take collapses the windows inside it to a confidence of
#: zero, so one continuous recording comes back as two spans carrying the SAME offset —
#: measured, and downstream that reads as a stop/restart that never happened. Two spans
#: agreeing on the offset across a short gap are one take with an unusable patch: the
#: camera kept rolling in sync, only the audio stopped saying so.
#:
#: The bound is what stops that from becoming a lie. A clip that records the song, then
#: five minutes of something else, then the song again at the same offset would otherwise
#: be merged into one span claiming those five minutes align. Past this gap the two are
#: reported separately, because correlation cannot tell "quiet" from "different material"
#: and inventing an answer is worse than reporting both.
SPAN_MERGE_GAP_S: float | None = None


@dataclass(frozen=True)
class ClipAlignment:
    """Where one clip sits on a reference (song) timeline.

    Attributes:
        index: The clip's position in the input sequence.
        offset_s: Reference-time where the clip's audio begins (may be negative).
        confidence: Normalized cross-correlation coefficient in ``[0, 1]``.
        duration_s: The clip's own duration (seconds).
        coverage: ``(start_s, end_s)`` — the clip's span **intersected with the
            reference timeline** ``[0, reference_duration]``. When the clip does not
            overlap the reference at all this is a degenerate ``(t, t)`` and
            :attr:`overlaps` is False; callers building an edit must skip those.
        overlaps: Whether the clip intersects the reference timeline at all. A clip that
            does not is still RETURNED, with its measured offset and confidence — see
            :func:`align_clips_to_reference` for why it is not dropped.
        support: How much of the clip's analysis windows' own evidence — before the
            consensus vote — reaches :attr:`offset_s`, in ``[0, 1]``: "how much of this
            clip agrees that this is where it goes". It is a different question
            from :attr:`confidence`, which asks only how well the clip matches at the
            offset reported, and it is the one that catches the two failures a
            coefficient cannot: a clip that matches beautifully **somewhere else too**
            (repetitive music), and a clip only PART of which is the reference at all.

            **A graded tally, not a headcount** (issue #45). A window whose own argmax
            landed on :attr:`offset_s` contributes a full 1.0; a window that merely put
            it on its ballot — could not separate it from its own answer, or had it
            nominated by the other feature — contributes up to
            :data:`BALLOT_VOTE_WEIGHT`, scaled by how close it scored; a window that
            never considered it contributes nothing. Ballot mentions alone cannot carry
            the tally past ``0.5``, so **``support > 0.5`` means at least one independent
            window reached this offset unaided** — which is exactly what the whole number
            used to mean, now the top half of a range rather than all of it.

            **A threshold carried over from before the change no longer means what it
            meant.** Every value in ``(0, 0.5]`` is reachable with zero windows having
            found the offset unaided, so a gate at 0.25 now says "some of the clip's
            evidence mentions this offset", not "a quarter of the windows located it
            themselves". A caller who meant the latter gates at ``> 0.5``, and that is
            not a cosmetic re-tune: measured at the default window, an exactly tiling
            reference — where the offset is a free choice among nine — now lands just
            *below* 0.5 and a reference that is one half twice lands just *above* it, so
            a gate at 0.5 separates them and a gate at 0.25 passes both.

            **The change is a no-op only where the argmax was already decisive.** A long
            clip at the default 20 s window is unchanged on a reference that does not
            repeat, because every window got there on its own and there is nothing to
            add. On a repeating reference the same long clip moves — measured, a 60 s
            clip at ``window_s=20`` went from a headcount of 0.00 to about 0.50. Length
            was never the thing that made the old number safe; an undisputed argmax was.

            The reason it is graded: an argmax is a real opinion at a 20 s window and
            close to a coin flip at 4 s, so a bare headcount got *less* confident exactly
            as fitting the window to the clip (issue #41) made the estimator *more*
            reliable — backwards, for a trust gate. Measured on real cross-device
            material, 21 alignments that were ALL correct reported 0.00-1.00 with six at
            0.00: the vote landed on the right offset while no individual window's argmax
            agreed.

            Measured on three phone recordings of one commercial track, the whole-clip
            argmax was 83 s, 174 s and 83 s wrong while its coefficient looked ordinary;
            consensus support was 10/24, 17/37 and 45/61 and pointed at offsets three
            independent methods then confirmed to within 40 ms (issue #30).
            **``None`` when it was not measured** — with ``consensus=False``, for a clip
            short enough that the window in force gives it fewer than two INDEPENDENT
            looks — which takes about ``window_s + hop_s`` of clip, not one window's worth,
            so at the default grid a 26 s clip had no support either; with
            ``window_s=None`` that threshold moves down to one floor-window plus its hop
            (see :func:`_clip_window_and_hop`) — and when the windows overlap too heavily
            to be separate opinions (:data:`MAX_SUPPORT_OVERLAP`). None of those has a
            second opinion to compare against, and a support of 1.0 there would be a
            unanimous vote of one: a number that VOUCHES for an offset nothing
            corroborated. That is not hypothetical — a 15 s clip truly at offset 30.0,
            against a reference that is one half twice, comes back at offset 75.0 with
            confidence 0.979; and on real material a 10 s clip at ``window_s=9.5,
            hop_s=0.5`` gave two 95%-overlapping windows that agreed on an offset 102 s
            wrong, which the tally reported as 1.00 before those windows were excluded.
            ``None`` says "not measured", which a caller can fall back from.

            **It is relative to ``window_s``, so a gate on it is too.** Support asks how
            much independent evidence agreed, and a shorter window is a weaker opinion.
            Grading the tally softens that — a short window's near-miss is now worth
            something rather than nothing — but it does not remove it: measured on the
            same three correct cross-device alignments, the headcount this replaced read
            0.45/0.64/0.73 at ``window_s=20`` and 0.19/0.16/0.23 at ``window_s=5``. It is
            deliberately not normalised — dividing by something to make the numbers look
            stable would invent a statistic — so a caller that changes ``window_s`` must
            revisit its threshold, and a caller comparing two clips must compare them at
            the same window. With ``window_s=None`` the window is fitted to each clip, so
            two clips of different lengths are NOT at the same window unless both are long
            enough to sit at the default; a caller that ranks clips by support should pass
            an explicit ``window_s`` to put them back on one scale, or read
            :attr:`window_s` and scale its threshold per clip.

            It says how much evidence reaches this offset and nothing about how much
            reaches somewhere else — see :attr:`margin` for that half.
        margin: How far :attr:`support`'s tally for this offset sits above the tally for
            the best DIFFERENT offset — the best one outside ``offset_tolerance_s`` of
            it — or ``None`` when there was nothing to measure. Support says how much of
            the clip's own evidence reaches this offset; margin says **whether anything
            else has an equal claim** (issue #47). A coefficient cannot answer that: on
            real material the wrong offset's peak scored 0.987-0.993 of the right one's,
            and support cannot answer it either, because an ambiguous tiling lands mid-
            scale by construction — every alias is genuinely well supported.

            **Scale.** It is a difference of two support tallies, read on
            :attr:`support`'s scale and in its units. ``1.0`` means every independent
            window found this offset unaided and no window put any other offset forward
            at all. ``0.0`` means some rival offset is backed by exactly as much of the
            clip's evidence as the answer is.

            **A wide margin is not a claim of sub-tolerance precision.** Differences
            smaller than ``offset_tolerance_s`` are one hypothesis by construction — the
            same equivalence the vote groups ballots by — so a candidate that close is
            the answer, not a runner-up, and never subtracts. Measured at the default
            tolerance: a rival 0.20 s away leaves the margin at 1.000, and the same
            rival at 0.30 s takes it to 0.505. The field says nothing has an equal claim
            *at a different offset*; how sharply the offset itself is located is
            ``offset_tolerance_s``, and it is the caller's to set. Where no window nominated any rival the
            runner-up's tally is ``0.0``, so the margin is simply the support — nothing
            else has any claim. It can come back slightly NEGATIVE: the offset is chosen
            by the vote's headcount over every window, while the tally is graded and read
            over the independent subset only (:data:`MAX_SUPPORT_OVERLAP`), so the two
            need not rank identically. A negative margin says the tally actually prefers
            somewhere else, which is the strongest "do not trust this" this object
            carries.

            **How to combine it with support.** They answer different questions, and a
            trust gate wants both:

            - high support, wide margin — the evidence points here, and nowhere else;
            - high support, margin near zero — the clip fits here *and fits somewhere
              else exactly as well*. An exactly tiling reference is the pure case: every
              alias draws the same tally, and no estimator can say which one was meant;
            - low support, wide margin — thin evidence, undisputed. A short clip on a
              repeating bed lands here (issue #45): no window's argmax resolved the tile,
              so the tally sits near :data:`BALLOT_VOTE_WEIGHT`, and yet no other offset
              drew any of it. This is the row a support-only gate refuses and should not;
            - low support, margin near zero — nothing is known.

            So **support is a floor and margin is a separator**. Measured on the real
            cross-device material this was built for — 24 correct alignments against 6
            pure-noise clips — the separator is doing nearly all of the work:
            ``support > 0.5`` passed 18 of 24 and refused all 6; adding
            ``and margin > 0`` to it changed nothing (margin never refused what support
            passed); and **``margin > 0`` alone passed all 24 and still refused all 6**,
            recovering every correct short clip the support floor was turning away.
            Five of the six noise clips came back NEGATIVE (-0.167 to -0.317) and no
            correct alignment did, so a negative margin is worth reading as a refusal
            rather than merely as a failure to vouch.

            **Keep a floor under it anyway.** The sixth noise clip scored exactly
            ``+0.000`` — refused by an exact tie, which a different draw does not
            guarantee — and two correct alignments passed at ``+0.014`` and ``+0.029``
            on a support of 0.34-0.35, which is the "thin evidence, undisputed" row with
            nothing beneath it. A conjunction such as ``support > 0.25 and margin > 0``
            scored the same 24/6 there while giving five of the six noise clips a second,
            independent reason to fail. Thirty cases is encouraging and is not proof.

            Margin is relative to :attr:`window_s` for exactly the reason support is —
            it is built from the same tally — so a threshold on it moves with the window
            the same way.

            **``None`` means unmeasured**, never manufactured, on the same quorum as
            :attr:`support` (:data:`MIN_WINDOWS_FOR_SUPPORT`): one independent look has
            no runner-up to be ahead of, and a number invented there would VOUCH.
        window_s: The analysis window this clip's vote was actually held at, in seconds —
            **the scale :attr:`support` is expressed on**, reported because since the
            window is fitted to the clip it is no longer something the caller can infer
            from its own arguments. ``None`` when no vote was held (``consensus=False``),
            for the same reason ``support`` is.

            Read it whenever you gate on support. A fixed threshold applied across clips
            measured at different windows compares numbers that are not comparable:
            measured on real cross-device material, 21 alignments that were all CORRECT
            reported the bare argmax headcount from 0.00 to 1.00 (which is why that tally
            is now graded — :data:`BALLOT_VOTE_WEIGHT`) depending mostly on clip length,
            because a 4 s window on a 12 s clip is a weaker opinion than a 20 s window on
            a 60 s one. With this field a caller can scale its gate to the window, or
            decline to gate when the window came out small — what it cannot do is read
            0.33 and 0.75 as if they answered the same question.

            **Necessary, and since #45 no longer sufficient.** A rule of the form "below
            the default window, treat the value as unmeasured" was a complete defence
            while support was an argmax headcount, because that was the only axis that
            moved it. It is not any more: what a repeating reference does to the tally it
            does at ``window_s=20`` too — a 60 s clip there went from 0.00 to about 0.50
            on a bed that tiles. So reading this field tells a caller how strong each
            opinion was; it does not put the number back on a scale a pre-#45 threshold
            was calibrated against. Nothing does. A threshold calibrated on the headcount
            has to be re-measured against this statistic, not re-pointed at it.

            **And such a guard should be REMOVED by this change, not kept alongside a new
            threshold** — measured, not reasoned. A rule that declines to trust support
            below the default window has to route those clips somewhere, and the only
            thing left is :attr:`confidence`. That fallback is weaker than the statistic
            the guard was protecting against: on six pure-noise clips at a fitted 6.67 s
            window, confidences of 0.017-0.173 passed a consumer's floor of 0.1 for
            **four of the six**, while graded support at ``> 0.5`` refused all six.
            Grading is what makes support usable at a fitted window in the first place,
            so a window guard in front of it now costs more than it saves.
        hop_s: The step between those windows — the other half of the grid, reported for
            the same reason and ``None`` in the same cases. Support counts windows that
            are separated enough to be second opinions
            (:data:`MAX_SUPPORT_OVERLAP`), so which windows were *eligible* to agree
            depends on the hop as much as on the window: a support figure is reproducible
            from ``(window_s, hop_s)`` and not from either alone.
    """

    index: int
    offset_s: float
    confidence: float
    duration_s: float
    coverage: tuple[float, float]
    overlaps: bool = True
    support: "float | None" = None
    margin: "float | None" = None
    window_s: "float | None" = None
    hop_s: "float | None" = None


def _window_samples(seconds: float, sample_rate: int) -> int:
    """Samples in a window of ``seconds`` — :func:`_window_offsets`' own rounding.

    Duplicated deliberately rather than shared: the guard must decide on the SAME
    integer the grid will be built from, and a guard that rounds even slightly
    differently is a guard that disagrees with the thing it guards.
    """
    return max(1, int(round(seconds * sample_rate)))


def _max_supportable_window_s(
    clip_duration_s: float,
    sample_rate: int,
    *,
    max_overlap: float = MAX_SUPPORT_OVERLAP,
) -> float:
    """The widest window at which a clip of this length still holds a SECOND look.

    Roughly ``clip_duration_s / (1 + max_overlap)``, and it does NOT depend on ``hop_s`` —
    which is the whole subtlety. A hop-based bound (``duration - hop``) is wrong in both
    directions: at ``window=11, hop=1`` on a 15 s clip it accepts a call that returns
    ``support=None``, and at ``window=6, hop=12`` it refuses one that measures
    ``support=1.0``.

    The reason the hop drops out is :func:`_window_offsets`. Its regular grid stops at the
    last start that fits, and it then APPENDS a tail window starting at
    ``len(clip) - n_win`` whenever the grid did not already reach the end — so whatever
    the hop, the last window begins exactly one window's length from the clip's end.
    :func:`_independent_windows` keeps a second window once it has moved on by
    ``1 - max_overlap`` of a window, so a quorum exists iff
    ``clip_duration - window >= max_overlap * window``. The hop decides how MANY looks
    there are and how finely a boundary is placed, never whether a second independent one
    can exist at all.

    **Computed in SAMPLES, which is not a detail.** The grid rounds the window to whole
    samples before any of this happens, and at the seconds bound that rounding can go the
    wrong way by half a sample: on a 10 s clip at 16 kHz the exact ``10 / 1.5`` rounds to
    106667 samples where 106666 is the most that fits, so a caller retrying at the
    ceiling this function is FOR would land straight back in the exception. Taking the
    floor in samples makes the bound admissible by construction wherever one exists,
    which is the only thing that makes it worth reporting. The exception is a degenerate
    clip of a sample or two, which holds no second look at ANY window: the result is
    floored at one sample there and is a lower bound rather than a promise.
    """
    clip_samples = int(round(clip_duration_s * sample_rate))
    max_window_samples = int(clip_samples / (1.0 + max_overlap))
    return max(1, max_window_samples) / sample_rate


def _holds_a_second_look(
    clip_duration_s: float,
    window_s: float,
    sample_rate: int,
    *,
    max_overlap: float = MAX_SUPPORT_OVERLAP,
) -> bool:
    """Whether this window leaves the clip a second INDEPENDENT look.

    :func:`_independent_windows`' own test, on the two windows that always exist: the one
    at the start and the one ending at the clip's end.
    """
    clip_samples = int(round(clip_duration_s * sample_rate))
    window_samples = _window_samples(window_s, sample_rate)
    return clip_samples - window_samples >= max_overlap * window_samples


def _clip_window_and_hop(
    clip_duration_s: float,
    sample_rate: int,
    *,
    window_s: "float | None",
    hop_s: "float | None",
    clip_index: "int | None" = None,
    default_window_s: float = SPAN_WINDOW_S,
    target_windows: int = MIN_WINDOWS_FOR_SUPPORT_TARGET,
    min_frames: int = ADAPTIVE_WINDOW_MIN_FRAMES,
    envelope_hop: int = ENVELOPE_HOP,
    hop_ratio: float = ADAPTIVE_HOP_RATIO,
) -> "tuple[float, float]":
    """The window and hop to measure ONE clip with — ``None`` means "fit them to it".

    ``window_s = min(default, clip_duration_s / target_windows)``, floored at the shortest
    window the onset envelope can carry (:data:`ADAPTIVE_WINDOW_MIN_FRAMES` frames of
    :data:`ENVELOPE_HOP`), with a hop of :data:`ADAPTIVE_HOP_RATIO` of whatever window
    comes out. An explicit ``window_s`` is never overridden — the caller has said what a
    window means for their material and that outranks any rule here — and an explicit
    ``hop_s`` likewise.

    **Why a clip's own length decides.** A clip too short for the window in force has no
    second opinion to arbitrate: the offset it returns is whatever one correlation says,
    and :func:`_support_fraction` honestly reports ``None`` for it. Measured on a real
    cross-device shoot (issue #41), a 10 s clip against a 250 s reference came back 102 s
    from the truth at confidence 0.834 that way, while the same clip at every window from
    3 s to 6 s was right. The clip's length is known before any correlation runs, so this
    is decidable up front, and the vote the estimator already knows how to hold is simply
    made available to short clips too.

    **"Too short" is about independent LOOKS, not about fitting in a window.** Support
    counts windows separated by at least :data:`MAX_SUPPORT_OVERLAP` of their length, so
    the default grid needs roughly ``window_s + hop_s`` — 30 s — before a clip has a second
    look at all: measured, 22, 24 and 26 s clips against a 90 s reference all reported
    ``None`` at the default, and the 22 s one landed 64 s from the truth at confidence
    0.279 where ``window_s=10`` was right to 3 ms. Fitting the window to the clip covers
    every one of them; a rule written on "shorter than one window" would have covered none.
    Below one floor-window plus its hop, no windowing can hold a quorum and ``None``
    remains the answer.

    **It moves ``support``, not only the offset.** Support is relative to ``window_s``
    (see :attr:`ClipAlignment.support`), so a clip measured at an adapted 3.3 s window
    reports a smaller number than the same clip would at 20 s — the same three correct
    real alignments read 0.45/0.64/0.73 at 20 s and 0.19/0.16/0.23 at 5 s. A short clip had
    no support at all before, so nothing is being reinterpreted; but a caller gating on
    support must know that the number arrives on the clip's scale, not the default's.

    **An EXPLICIT window too wide for the clip is refused, not honored** (issue #43).
    Above :func:`_max_supportable_window_s` the clip cannot hold two INDEPENDENT windows
    (:func:`_holds_a_second_look`),
    so the vote the caller asked for cannot be held and ``support`` comes back ``None`` —
    which reads exactly like a clip too short to support at any window. Rather than
    degrade silently, this raises :class:`~mixing.errors.WindowTooWideForClip`, naming the
    clip, its duration and the largest window that would still hold a quorum. Only an
    explicit window is refused: the ``None`` path above fits the window to the clip and,
    at the floor, may legitimately land on a single window, which is the honest answer for
    a clip that short and not a caller's mistake.
    """
    explicit_window = window_s is not None
    if window_s is None:
        floor_s = min_frames * envelope_hop / sample_rate
        window_s = min(default_window_s, max(floor_s, clip_duration_s / target_windows))
    if hop_s is None:
        hop_s = window_s * hop_ratio
    if explicit_window and not _holds_a_second_look(
        clip_duration_s, window_s, sample_rate
    ):
        raise WindowTooWideForClip(
            clip_index=clip_index,
            clip_duration_s=clip_duration_s,
            window_s=window_s,
            hop_s=hop_s,
            max_window_s=_max_supportable_window_s(clip_duration_s, sample_rate),
        )
    return float(window_s), float(hop_s)


def align_clips_to_reference(
    reference_audio: AudioSource,
    clips: "Sequence[AudioSource]",
    *,
    reference_duration: float | None = None,
    sample_rate: int = 16000,
    min_overlap_ratio: float = 0.5,
    feature: str = "envelope",
    consensus: bool = True,
    window_s: "float | None" = None,
    hop_s: "float | None" = None,
    offset_tolerance_s: float = SPAN_OFFSET_TOLERANCE_S,
    near_tie_ratio: float = NEAR_TIE_RATIO,
) -> list[ClipAlignment]:
    """Align a SET of clips to one reference — the multi-device / multicam primitive.

    Aligns each clip against ``reference_audio`` (e.g. the clean song) and returns its
    offset, a scale-invariant confidence, and its **coverage clamped to the reference
    timeline** — so a downstream editor gets valid spans and never references a time the
    reference does not cover. Preserves the original ``index`` so callers can map results
    back to inputs.

    **Every clip gets a record.** A clip with no temporal overlap is returned with
    ``overlaps=False`` rather than omitted, because omission is how a source silently leaves
    the addressable set: a caller that persists this list as *the* alignment artifact ends
    up with material it can no longer reference, name, or explain — the file is still there,
    but nothing downstream can point at it. Selecting what goes into an edit is a matter of
    *referencing* sources and intervals; a source must never disappear from what can be
    referenced as a side effect of being measured. Callers building an edit filter on
    ``overlaps``; callers reporting to a human show all of them, with the reason.

    Args:
        reference_audio: The signal every clip is aligned to (the song).
        clips: The clip audio sources (paths, arrays, or ``AudioSegment``\\ s).
        reference_duration: The reference timeline length (seconds); computed from
            ``reference_audio`` when omitted.
        sample_rate: Analysis sample rate (mono).
        min_overlap_ratio: Passed through to the alignment (see
            :func:`find_audio_offset_detailed`).
        feature: Similarity feature the alignment is measured on — see
            :data:`ALIGNMENT_FEATURES`. Under ``consensus`` it chooses the OFFSET and not
            only the confidence: ``'envelope'`` lets the onset envelope nominate lags the
            waveform never offers, which on cross-device material is the only way the true
            offset reaches a window's ballot at all (issue #30).
            Defaults to ``'envelope'`` **because this function's whole purpose is the
            cross-device case**, and a raw-waveform coefficient is not a usable trust gate
            there: two microphones in a room are not sample-correlated even when the
            alignment is exact. Measured on a real 6-device shoot, the waveform coefficient
            scored provably-correct alignments at 0.064-0.148 — below any threshold a
            caller would sensibly set — while the envelope scored them 0.441-0.634 and a
            genuine non-match at 0.102. Pass ``'waveform'`` when the clips come from the
            SAME source as the reference (e.g. verifying an export against its master),
            where sample correlation is meaningful and gives finer confidence resolution.
        consensus: Estimate the offset by putting the clip's analysis windows to a vote
            rather than by one argmax over the whole clip (issue #30). **On by default,
            because the whole-clip argmax is measurably wrong on repetitive material**:
            it minimises a correlation whose peaks are near-tied at musical periods, so
            it returns whichever repeat won a coin flip, and its coefficient does not
            drop when it does. A spurious peak lands at a different lag in every window
            while the true offset is the one they share, so the windows' agreement is
            what separates them — and how much of the clip agrees is reported as
            :attr:`~ClipAlignment.support`, with how far that is ahead of the runner-up
            offset as :attr:`~ClipAlignment.margin`. ``False`` restores the single
            whole-clip correlation exactly — same offset, same confidence, and
            ``support=None``, ``margin=None`` because nothing was put to a vote. It
            is cheaper (one correlation instead of one per window) and is the right
            choice only when the reference is known not to repeat.
        window_s: Analysis window for the vote. Ignored when ``consensus`` is False.
            ``None`` (the default) **fits the window to each clip** — see
            :func:`_clip_window_and_hop`: ``min(20 s, clip_duration / 3)``, floored at
            what the onset envelope can carry, so a clip too short to hold three default
            windows still gets a vote and a measured ``support`` instead of one
            uncorroborated correlation (issue #41). Whatever window each clip ends up
            measured at is reported back as :attr:`ClipAlignment.window_s`, because it is
            the scale its ``support`` is on. A clip of three default windows or
            more is measured at :data:`SPAN_WINDOW_S` exactly, so nothing about a long
            clip's answer moves. Pass a number to fix the window yourself — an explicit
            value is never overridden.

            **An explicit window a clip cannot hold is REFUSED** with
            :class:`~mixing.errors.WindowTooWideForClip` (issue #43). The bound is
            ``clip_duration / (1 + MAX_SUPPORT_OVERLAP)`` and does NOT involve ``hop_s``
            (:func:`_max_supportable_window_s`): above it no second INDEPENDENT window
            exists, so the vote asked for cannot be held and the result would be one
            whole-clip correlation reporting ``support=None`` — the same thing a clip too
            short to support at any window reports, with nothing to tell the two apart.
            The error names the clip, its duration and the largest window that still
            leaves a second look — measurable for any clip long enough to be worth
            aligning, though a degenerate clip of a sample or two has no such window at
            all. To measure such a clip, pass that window, pass ``window_s=None`` to fit
            the window to each clip, or pass ``consensus=False`` to ask for the single
            correlation outright.
        hop_s: Step between those windows. Ignored when ``consensus`` is False. ``None``
            (the default) is :data:`ADAPTIVE_HOP_RATIO` of whatever window is in force —
            half of it, the default pair's own ratio — so an adapted grid keeps the
            default grid's shape.

            **Passing ``window_s`` alone now moves the hop too**, and that is a change:
            before the window was fitted, an unpassed ``hop_s`` was a flat 10 s, so
            ``window_s=5`` meant the pair ``(5, 10)`` — a hop twice the window, which
            skips over half the clip and was never anyone's intent. It now means
            ``(5, 2.5)``. Nothing else about an explicit ``window_s`` moved, but a
            ``support`` measured at ``window_s=5`` on an earlier version is not
            reproducible here without passing ``hop_s=10`` alongside it. The pair each
            clip was actually measured at comes back as :attr:`ClipAlignment.window_s`
            and :attr:`ClipAlignment.hop_s`.
        offset_tolerance_s: How far two windows' offsets may differ and still count as
            the same answer.
        near_tie_ratio: How close a rival peak must score to a window's best one to join
            the vote — see :data:`NEAR_TIE_RATIO`.

    Returns:
        A list of :class:`ClipAlignment`, in input order (minus dropped clips).

    Raises:
        ValueError: ``feature`` is not one of :data:`ALIGNMENT_FEATURES`, or
            ``near_tie_ratio`` is negative.
        ~mixing.errors.WindowTooWideForClip: ``consensus`` is on and an EXPLICIT
            ``window_s`` is wider than some clip can hold a second independent look at
            (issue #43). The error names that clip, its duration and the largest window
            that would work; it subclasses ``ValueError``. Never raised on the
            ``window_s=None`` path, which fits the window to each clip.
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    if near_tie_ratio < 0:
        raise ValueError(f"near_tie_ratio must not be negative, got {near_tie_ratio}")
    ref = _load_mono_samples(reference_audio, sample_rate)
    ref_dur = (
        reference_duration if reference_duration is not None else len(ref) / sample_rate
    )
    # The reference's envelope does not change between clips, let alone between windows.
    ref_env = onset_envelope(ref, sample_rate) if feature == "envelope" else None
    out: list[ClipAlignment] = []
    for i, clip in enumerate(clips):
        query = _load_mono_samples(clip, sample_rate)
        if consensus:
            # Per clip, not per call: the windowing is fitted to the clip being measured,
            # so a short clip in a set does not inherit a long one's window.
            clip_window_s, clip_hop_s = _clip_window_and_hop(
                len(query) / sample_rate,
                sample_rate,
                window_s=window_s,
                hop_s=hop_s,
                clip_index=i,
            )
            offset_s, coeff, support, margin = _consensus_alignment(
                ref,
                query,
                sample_rate,
                window_s=clip_window_s,
                hop_s=clip_hop_s,
                feature=feature,
                min_overlap_ratio=min_overlap_ratio,
                near_tie_ratio=near_tie_ratio,
                offset_tolerance_s=offset_tolerance_s,
                ref_envelope=ref_env,
            )
        elif feature == "envelope":
            lag, coeff = _envelope_then_waveform(
                ref,
                query,
                sample_rate,
                min_overlap_ratio=min_overlap_ratio,
                ref_envelope=ref_env,
            )
            # Nothing was put to a vote, so neither support, nor the margin over a
            # runner-up, nor the window they would be relative to exists — and support
            # is never 1.0 here.
            offset_s, support, margin = lag / sample_rate, None, None
            clip_window_s = clip_hop_s = None
        else:
            lag, coeff = _normalized_xcorr(
                ref, query, min_overlap_ratio=min_overlap_ratio
            )
            offset_s, support, margin = lag / sample_rate, None, None
            clip_window_s = clip_hop_s = None
        dur_s = len(query) / sample_rate
        start = max(0.0, offset_s)
        end = min(ref_dur, offset_s + dur_s)
        overlaps = end > start
        out.append(
            ClipAlignment(
                index=i,
                offset_s=offset_s,
                confidence=coeff,
                duration_s=dur_s,
                coverage=(start, end) if overlaps else (start, start),
                overlaps=overlaps,
                support=support,
                margin=margin,
                window_s=clip_window_s,
                hop_s=clip_hop_s,
            )
        )
    return out


def _independent_windows(
    windows: "Sequence[_WindowMeasurement]",
    *,
    max_overlap: float = MAX_SUPPORT_OVERLAP,
) -> "list[_WindowMeasurement]":
    """The subset of ``windows`` that are second opinions rather than the same look twice.

    Greedy from the first window: keep a window only once it has moved on by at least
    ``1 - max_overlap`` of the previously kept window's own length. Overlapping windows
    are not noise to be removed — they are how a span boundary is located to better than
    one window — but they are not INDEPENDENT, and support is a count of independent
    agreement (see :data:`MAX_SUPPORT_OVERLAP`).
    """
    kept: "list[_WindowMeasurement]" = []
    for window in windows:
        if not kept:
            kept.append(window)
            continue
        previous = kept[-1]
        stride = (previous.clip_end_s - previous.clip_start_s) * (1.0 - max_overlap)
        if window.clip_start_s - previous.clip_start_s >= stride:
            kept.append(window)
    return kept


def _window_agreement(
    window: "_WindowMeasurement",
    offset_s: float,
    *,
    offset_tolerance_s: float,
    ballot_weight: float = BALLOT_VOTE_WEIGHT,
) -> float:
    """How much ONE window's evidence says ``offset_s``, on a scale of ``[0, 1]``.

    Three grades, and the middle one is the whole point (issue #45):

    - **1.0** — the window's own argmax (:attr:`_WindowMeasurement.vote_offset_s`) is
      there. It found the offset unaided; nothing corroborates it more strongly than
      that.
    - **``ballot_weight * score / best``** — the offset is on the window's ballot but was
      not its answer: the window could not separate it from its own best-scoring
      candidate (:data:`NEAR_TIE_RATIO`), or another feature nominated it. The ratio is
      **within this window and uncalibrated** — both scores are
      :func:`_dual_confidence` values, so it says how this candidate ranked against the
      window's own best and nothing about either in absolute terms.
    - **0.0** — the window never considered it. No vote can be read out of an offset
      nobody put forward.

    Monotone in the evidence by construction: strengthening what a window says about
    ``offset_s`` (unmentioned → mentioned → mentioned higher → argmax) never lowers this
    window's grade, so **for a fixed set of independent windows** it never lowers the
    support tally either. That qualifier is load-bearing: adding a window changes which
    windows :func:`_independent_windows` keeps, and the greedy stride can displace a
    full-credit window in favour of the newcomer — so an extra agreeing window can lower
    the mean even though no window's own grade fell.

    ``ballot_weight=0.0`` reduces this to the bare argmax headcount that
    :func:`_support_fraction` used to be — the before, available as a measurement.
    """
    if abs(window.vote_offset_s - offset_s) <= offset_tolerance_s:
        return 1.0
    best = max((score for _, score in window.candidates), default=0.0)
    if best <= 0:
        return 0.0
    on_ballot = [
        score
        for candidate_offset, score in window.candidates
        if abs(candidate_offset - offset_s) <= offset_tolerance_s
    ]
    if not on_ballot:
        return 0.0
    return ballot_weight * max(on_ballot) / best


def _support_fraction(
    windows: "Sequence[_WindowMeasurement]",
    offset_s: float,
    *,
    offset_tolerance_s: float,
) -> "float | None":
    """How much of the INDEPENDENT windows' evidence reaches ``offset_s``, or ``None``.

    The mean of :func:`_window_agreement` over the independent windows: a window that
    reached ``offset_s`` unaided contributes 1.0, one that only put it on its ballot
    contributes up to :data:`BALLOT_VOTE_WEIGHT`, one that never considered it
    contributes nothing.

    **This was a bare argmax tally and is now a graded one** (issue #45). The old
    definition asked "how many windows' independent argmax landed here", which is a real
    question at a 20 s window and close to a coin flip at 4 s — so once the window was
    fitted to the clip (issue #41) the estimator got *more* reliable as its confidence
    statistic got *less*, which is the wrong way round for a trust gate. Measured on real
    cross-device material, six of 21 alignments that were all CORRECT reported 0.00.
    Grading the tally keeps the old question answerable — ballot mentions alone cannot
    reach past :data:`BALLOT_VOTE_WEIGHT`, so ``support > 0.5`` still means some window
    got there on its own — while giving the windows that nearly got there a number
    instead of a zero.

    ``None`` means "not measured", and it is not spelled ``1.0`` on purpose: a fraction
    computed over fewer than :data:`MIN_WINDOWS_FOR_SUPPORT` independent looks is a
    unanimous vote of one, and a manufactured 1.0 VOUCHES for whatever it is attached to.

    Counted on each window's own candidates, never on what the consensus assigned it —
    the assignment agrees with itself by construction.
    """
    counted = _independent_windows(windows)
    if len(counted) < MIN_WINDOWS_FOR_SUPPORT:
        return None
    return _graded_tally(counted, offset_s, offset_tolerance_s=offset_tolerance_s)


def _graded_tally(
    counted: "Sequence[_WindowMeasurement]",
    offset_s: float,
    *,
    offset_tolerance_s: float,
) -> float:
    """The mean of :func:`_window_agreement` over ALREADY-independent windows.

    Split out from :func:`_support_fraction` because the same tally has to be read at
    more than one offset: :func:`_support_margin` compares the winner's against every
    rival's, and a margin between two numbers computed different ways would not be a
    margin. The quorum rule and the ``None`` live with the caller — this is arithmetic
    over the windows it is handed, and it assumes they were already filtered by
    :func:`_independent_windows`.
    """
    # The weight is looked up here rather than left to the helper's default so that
    # setting it to 0 — which is exactly the argmax headcount this replaced — is a thing a
    # characterization test can do, the way `near_tie_ratio=0.0` restores the pre-#30
    # argmax path.
    agreement = [
        _window_agreement(
            w,
            offset_s,
            offset_tolerance_s=offset_tolerance_s,
            ballot_weight=BALLOT_VOTE_WEIGHT,
        )
        for w in counted
    ]
    return float(np.mean(agreement))


def _rival_offsets(
    counted: "Sequence[_WindowMeasurement]",
    offset_s: float,
    *,
    offset_tolerance_s: float,
) -> "list[float]":
    """The DISTINCT offsets these windows put forward that are not ``offset_s``.

    One representative per cluster: a candidate within ``offset_tolerance_s`` of one
    already collected is the same hypothesis seen from another window — the same
    equivalence :func:`_consensus_choice` groups ballots by — and counting a peak's
    shoulders as several rivals would understate the margin for no reason.

    Read off the windows' OWN ballots, so a rival exists only because some window
    nominated it. An offset nobody put forward is not a runner-up; it is a number.
    """
    rivals: "list[float]" = []
    for window in counted:
        for candidate_offset, _score in window.candidates:
            if abs(candidate_offset - offset_s) <= offset_tolerance_s:
                continue
            if any(abs(candidate_offset - r) <= offset_tolerance_s for r in rivals):
                continue
            rivals.append(candidate_offset)
    return rivals


def _support_margin(
    windows: "Sequence[_WindowMeasurement]",
    offset_s: float,
    *,
    offset_tolerance_s: float,
) -> "float | None":
    """How far ``offset_s``' tally sits above the best DIFFERENT offset's, or ``None``.

    :func:`_support_fraction` answers "how much of the clip's evidence reaches here".
    This answers the question that one cannot: **"and how much reaches somewhere else
    instead?"** (issue #47). Both are the same :func:`_graded_tally` over the same
    independent windows, so the difference is a difference of like for like.

    The runner-up is the best-tallying offset outside ``offset_tolerance_s`` of the
    winner — not the second-best voting group. The two differ exactly on the case that
    matters: on an exactly tiling reference every alias draws the same tally, and the
    aliases are what a caller needs to hear about. With no rival on any ballot the best
    rival tally is ``0.0``, so the margin is the winner's tally itself — nothing else
    has any claim at all.

    ``None`` follows the same quorum rule as support and for the same reason: one
    independent look cannot have a runner-up, and a manufactured margin VOUCHES.
    """
    counted = _independent_windows(windows)
    if len(counted) < MIN_WINDOWS_FOR_SUPPORT:
        return None
    tally = _graded_tally(counted, offset_s, offset_tolerance_s=offset_tolerance_s)
    rivals = _rival_offsets(counted, offset_s, offset_tolerance_s=offset_tolerance_s)
    best_rival = max(
        (
            _graded_tally(counted, rival, offset_tolerance_s=offset_tolerance_s)
            for rival in rivals
        ),
        default=0.0,
    )
    return tally - best_rival


def _consensus_alignment(
    ref: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    *,
    window_s: float,
    hop_s: float,
    feature: str,
    min_overlap_ratio: float,
    near_tie_ratio: float,
    offset_tolerance_s: float,
    ref_envelope: "tuple[np.ndarray, float] | None",
) -> "tuple[float, float, float | None, float | None]":
    """One ``(offset_s, confidence, support, margin)`` for a clip, by majority of windows.

    The whole-clip counterpart of what :func:`aligned_spans` does per span, for the
    caller who wants one number. The clip is cut into windows, each window's near-tied
    candidates vote (:func:`_consensus_choice`), and the offset the most windows landed
    on wins; the reported offset is the MEDIAN over that winning group, so the answer
    keeps sub-window precision rather than snapping to one window's estimate.

    ``support`` is counted on the windows' OWN candidates, not on what the vote assigned
    them — the vote's own output would agree with itself and say nothing. So it reads as
    "this much of the clip's own evidence reaches this offset", which drops both when the
    reference repeats and when only part of the clip is the reference at all. An
    independent argmax counts fully and a bare ballot mention counts up to
    :data:`BALLOT_VOTE_WEIGHT` (:func:`_window_agreement`).
    Under :data:`MIN_WINDOWS_FOR_SUPPORT` INDEPENDENT windows there is no second opinion
    to count — windows overlapping more than :data:`MAX_SUPPORT_OVERLAP` are one look
    read twice, not two — and it is ``None`` rather than 1.0 — see
    :attr:`ClipAlignment.support`.

    A clip shorter than one window is one window, and this returns exactly the offset
    and confidence the single whole-clip correlation would have.

    **Two offsets can genuinely tie.** A clip that fits equally often in two places
    splits its windows evenly and no evidence separates them; one is returned, with a
    support near 0.5 that is the honest report of a 50/50 — not a hedge, and not a claim
    to have chosen. ``margin`` is what makes that legible from the outside: it reads
    near zero there, and wide where the runner-up is genuinely behind
    (:func:`_support_margin`).
    """
    windows = _consensus_choice(
        _window_offsets(
            ref,
            clip,
            sample_rate,
            window_s=window_s,
            hop_s=hop_s,
            feature=feature,
            min_overlap_ratio=min_overlap_ratio,
            near_tie_ratio=near_tie_ratio,
            offset_tolerance_s=offset_tolerance_s,
            ref_envelope=ref_envelope,
        ),
        offset_tolerance_s=offset_tolerance_s,
    )
    if not windows:  # an empty clip has no windows and so no opinion
        return 0.0, 0.0, None, None
    offsets = np.array([w.offset_s for w in windows])
    coeffs = np.array([w.confidence for w in windows])
    agree = np.abs(offsets[:, None] - offsets[None, :]) <= offset_tolerance_s
    # Most windows wins; a tie goes to the group the EARLIER windows settled on. That is
    # the same continuity rule `_consensus_choice` breaks its own ties by, and it is one
    # rule on purpose: two passes tie-breaking on different principles could settle one
    # 50/50 clip two different ways inside a single call.
    counts = agree.sum(axis=1)
    winner = int(np.lexsort((np.arange(counts.size), -counts))[0])
    members = agree[winner]
    offset_s = float(np.median(offsets[members]))
    support = _support_fraction(
        windows, offset_s, offset_tolerance_s=offset_tolerance_s
    )
    margin = _support_margin(windows, offset_s, offset_tolerance_s=offset_tolerance_s)
    return offset_s, float(np.median(coeffs[members])), support, margin


@dataclass(frozen=True)
class AlignedSpan:
    """A maximal run of CLIP time that tracks the reference at one stable offset.

    ``align_clips_to_reference`` answers "where does this clip sit?" with a single
    number, which is the right answer only when the clip is one continuous take of the
    reference. A clip that was stopped and restarted, or that contains a chunk of
    something else, has no such number — and the single-offset model does not say so, it
    just returns the offset of whichever part correlated best and describes the rest of
    the clip wrongly.

    Attributes:
        clip_start_s: Where this span begins in the CLIP's own timeline.
        clip_end_s: Where it ends, in the clip's timeline.
        offset_s: Reference-time where this span's clip-time zero would fall — the same
            convention as :attr:`ClipAlignment.offset_s`, so
            ``reference_time = clip_time + offset_s`` holds inside the span.
        confidence: Representative confidence for the span (the median over its windows,
            not the max — a span is only as trustworthy as its typical window). It says
            **how well the clip matches where this span puts it**, and on a reference
            that repeats verbatim that is a question with several excellent answers —
            see :attr:`support`, and gate on both.
        support: How much of the span's INDEPENDENT windows' own evidence
            (:data:`MAX_SUPPORT_OVERLAP`) reaches this offset, in ``[0, 1]``, before the
            consensus vote and relative to ``window_s`` the same way
            :attr:`ClipAlignment.support` is — or ``None`` when the span was
            built from fewer than :data:`MIN_WINDOWS_FOR_SUPPORT` windows and there was
            therefore nothing to agree. A window that reached the offset unaided counts
            1.0 and one that only put it on its ballot counts up to
            :data:`BALLOT_VOTE_WEIGHT`, so ``support > 0.5`` still means some window got
            there on its own (issue #45). This is the number that knows about repetition.
            A reference with no repeats gives 1.0; a verse/chorus reference gives less,
            because some windows correlated just as well against the wrong chorus and
            only the crowd put them right; an exactly tiling reference gives very
            little, because there genuinely is no unique answer and the confidence alone
            would never say so. Low support with high confidence means "it fits here
            beautifully, and it would fit elsewhere too".

            **``None`` is not 1.0.** A span of one window cannot disagree with itself,
            so reporting 1.0 there would be a unanimous vote of one — a number that
            vouches for an offset nothing corroborated. ``None`` says "not measured",
            which a caller can fall back from; a manufactured 1.0 is what a caller
            trusts.
        margin: How far this span's :attr:`support` tally sits above the tally for the
            best DIFFERENT offset — the best one outside ``offset_tolerance_s`` of this
            span's — or ``None`` on the same quorum support is ``None`` on. Same
            statistic, same scale and same guidance as
            :attr:`ClipAlignment.margin`, which documents both in full: support is the
            floor ("this much of the span's evidence reaches here"), margin is the
            separator ("and nothing else has an equal claim"). Gate on both.

            It is the field that tells an exactly tiling reference from a merely
            repetitive one. Both report a middling support; only the tiling one reports
            a margin near zero, because there every alias is equally true and the span
            reports one of them without having chosen (issue #47).

            Two merged spans report the duration-weighted mean, and unmeasured plus
            measured is unmeasured — the same rule :attr:`support` merges under.
    """

    clip_start_s: float
    clip_end_s: float
    offset_s: float
    confidence: float
    support: "float | None" = None
    margin: "float | None" = None

    @property
    def duration_s(self) -> float:
        return self.clip_end_s - self.clip_start_s

    @property
    def reference_span(self) -> "tuple[float, float]":
        """This span's extent on the REFERENCE timeline."""
        return (self.clip_start_s + self.offset_s, self.clip_end_s + self.offset_s)


def aligned_spans(
    reference_audio: AudioSource,
    clip_audio: AudioSource,
    *,
    reference_duration: float | None = None,
    sample_rate: int = 16000,
    window_s: float = SPAN_WINDOW_S,
    hop_s: float = SPAN_HOP_S,
    min_confidence: float = SPAN_MIN_CONFIDENCE,
    offset_tolerance_s: float = SPAN_OFFSET_TOLERANCE_S,
    merge_gap_s: float | None = SPAN_MERGE_GAP_S,
    feature: str = "envelope",
    min_overlap_ratio: float = 0.5,
    near_tie_ratio: float = NEAR_TIE_RATIO,
) -> "list[AlignedSpan]":
    """The MAXIMAL spans of ``clip_audio`` that align to ``reference_audio``.

    The windowed counterpart of :func:`find_audio_offset_detailed`. Where that answers
    "where does this clip sit on the reference?" with one number, this answers "which
    PARTS of it sit there, and at what offset each?" — which is the only answerable
    question for a clip that was stopped and restarted, or that holds material the
    reference does not contain.

    A clip that is one continuous take **and that the correlation can verify
    throughout** returns exactly one span — the compatibility property a caller
    migrating from the single-offset model depends on. The qualification is load-bearing
    in two ways, both measured:

    - A stretch longer than ``merge_gap_s`` that nothing can verify (hard silence, a hand
      over the mic) splits the take into two spans reporting the SAME offset. Two
      same-offset neighbours are the *signal* that this happened, not a stop/restart.
    - A reference whose repeats are EXACT admits several true answers, and this reports
      one of them with a low :attr:`~AlignedSpan.support` rather than pretending to have
      chosen (see below).

    **Repetition is decided by consensus, not by argmax** (issue #30). A verse/chorus
    reference used to shatter one continuous take into 3 spans — one of them reporting a
    WRONG offset at 0.985 confidence — because each window picked its lag independently
    and a repeated chorus makes two peaks near-tied (measured within 0.2% here, and
    0.987-0.993 second-to-first on real music). So each window now puts its near-tied
    rivals forward instead of only its argmax, and takes whichever of them the most
    other windows can also read. A spurious peak lands at a different lag in every
    window; the true offset is the one they share. Measured on a repeated motif:
    verse/chorus 3 spans → 1, two identical halves 6 → 1, exact tiling 8 → 1, with the
    genuine stop/restart still at 2 and the non-repetitive take still at 1 — a window is
    never moved to a lag its own correlation did not already rate a near-tie, and a
    restarted take has no such lag near the old offset, so it departs freely.

    **What that costs you is told by ``support``, not by ``confidence``.** The
    confidence says how well the clip matches where the span puts it, and on repetitive
    material that question has several excellent answers. ``support`` — the fraction of
    the span's windows that found this offset unaided — is the one that knows: 1.0 on a
    reference with no repeats, less where the crowd had to intervene, and very little on
    an exactly tiling reference where there is genuinely no unique answer. **Gate on
    both.** High confidence with low support means "it fits here beautifully, and it
    would fit elsewhere too".

    **And ``support`` in turn cannot see a runner-up**, which is what
    :attr:`~AlignedSpan.margin` is for (issue #47). Support measures how much of the
    span's evidence reaches this offset; on an exactly tiling reference every alias is
    reached by the same evidence, so each of them would report a similar middling
    support and none of them is more true than the others. The margin — this offset's
    tally minus the best other offset's — reads near zero exactly there, and stays wide
    where a short clip's thin evidence is nonetheless undisputed. **Support is a floor;
    margin is a separator.**

    A span too short to hold a disagreement reports ``support=None``, meaning *not
    measured* — never 1.0. One window agrees with itself, and a unanimous vote of one
    is exactly the kind of number a caller would trust and should not.

    **Boundary resolution is ``window_s``, and no better.** A window is evidence that its
    whole extent aligns; a boundary falling inside a window degrades that window rather
    than locating itself within it. So a returned edge is accurate to roughly ±
    ``window_s``, which is ample for telling two takes apart and NOT enough to cut on.
    A caller who needs a tighter edge pays for it by shrinking ``window_s`` — the cost is
    linear in the number of windows. Stated here because a span that looks like a precise
    interval and is not is exactly the kind of number that gets used as one.

    **Decoded once.** Both signals are loaded through :func:`_load_mono_samples` a single
    time and the windows are slices of the resulting array. Re-decoding per window would
    be N times the cost *and* would reintroduce issue #25: pydub's rate conversion has no
    anti-alias filter, so a per-window decode path is a per-window chance to halve the
    confidence.

    Args:
        reference_audio: The signal to align within (e.g. the clean song).
        clip_audio: The recording to dissect.
        sample_rate: Analysis sample rate (mono).
        window_s: Analysis window — see the resolution note above.
        hop_s: Step between windows. Defaults to half a window, so every instant is
            covered twice and a boundary cannot hide between windows.
        min_confidence: A window must reach this for its span to exist at all.
        offset_tolerance_s: How far a window's implied offset may drift from its span's
            before it is treated as a different take.
        merge_gap_s: How long an unverified gap between two spans that AGREE on the
            offset may be and still be called one take. ``None`` (the default) uses one
            window. See :data:`SPAN_MERGE_GAP_S`.
        reference_duration: The reference timeline's length (seconds); computed from the
            decoded reference when omitted. Spans are trimmed to it, so a returned
            extent is always one the reference can honour. Same keyword, and the same
            purpose, as :func:`align_clips_to_reference`.
        feature: As :func:`align_clips_to_reference` — windowed, so it chooses the
            OFFSET and not only the confidence. Defaults to ``'envelope'`` for the same
            reason that function does: the cross-device case is what this is for.
        min_overlap_ratio: Passed through to the correlation.
        near_tie_ratio: How close a rival correlation peak must score to a window's best
            one to join the vote, as a fraction of that best score — see
            :data:`NEAR_TIE_RATIO`. ``0.0`` disables the consensus pass entirely and
            restores the per-window argmax this function shipped with, which is
            measurably wrong on repetitive material and is offered only for reproducing
            an older result.

    Returns:
        Spans in clip order, non-overlapping, and each lying entirely within
        ``[0, reference_duration]`` on the reference timeline. Empty when nothing in the
        clip aligns — which is a real answer, not a failure.

    >>> import numpy as np                                  # doctest: +SKIP
    >>> spans = aligned_spans(song, phone_recording)        # doctest: +SKIP
    >>> [(round(s.clip_start_s), round(s.offset_s)) for s in spans]   # doctest: +SKIP
    [(0, 12), (95, 240)]
    """
    if feature not in ALIGNMENT_FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; expected one of {ALIGNMENT_FEATURES}"
        )
    if hop_s <= 0 or window_s <= 0:
        raise ValueError(
            f"window_s and hop_s must be positive, got {window_s}, {hop_s}"
        )
    if hop_s > window_s:
        # Not a style rule: with a hop wider than the window, the clip time BETWEEN
        # consecutive windows is never looked at, yet a run spanning them reports one
        # span across the whole range — asserting alignment over instants no
        # correlation ever examined. `merge_gap_s` cannot bound it either, because the
        # gap is inside a run rather than between two spans.
        raise ValueError(
            f"hop_s ({hop_s}) must not exceed window_s ({window_s}): a larger hop "
            "leaves clip time unmeasured, and a span would then assert alignment over "
            "instants no window ever looked at."
        )
    if near_tie_ratio < 0:
        raise ValueError(f"near_tie_ratio must not be negative, got {near_tie_ratio}")
    ref = _load_mono_samples(reference_audio, sample_rate)
    clip = _load_mono_samples(clip_audio, sample_rate)
    measured = _consensus_choice(
        _window_offsets(
            ref,
            clip,
            sample_rate,
            window_s=window_s,
            hop_s=hop_s,
            feature=feature,
            min_overlap_ratio=min_overlap_ratio,
            near_tie_ratio=near_tie_ratio,
            offset_tolerance_s=offset_tolerance_s,
        ),
        offset_tolerance_s=offset_tolerance_s,
    )
    spans = _spans_from_windows(
        measured,
        min_confidence=min_confidence,
        offset_tolerance_s=offset_tolerance_s,
        merge_gap_s=window_s if merge_gap_s is None else merge_gap_s,
    )
    ref_dur = (
        len(ref) / sample_rate if reference_duration is None else reference_duration
    )
    return _clamp_to_reference(spans, ref_dur)


@dataclass(frozen=True)
class _WindowMeasurement:
    """What one analysis window has to say, INCLUDING the rivals it could not separate.

    ``candidates`` is ``((offset_s, confidence), ...)`` ordered by that confidence, and
    the first entry is the window's current answer — its own best-scoring nomination
    before :func:`_consensus_choice` runs, the crowd's choice after. ``vote_offset_s``
    keeps the window's own answer whatever happens to ``candidates``: it is its
    INDEPENDENT opinion, and the fraction of windows whose independent opinion matches
    the answer is what :attr:`AlignedSpan.support` reports.
    """

    clip_start_s: float
    clip_end_s: float
    candidates: "tuple[tuple[float, float], ...]"
    vote_offset_s: float

    @property
    def offset_s(self) -> float:
        return self.candidates[0][0]

    @property
    def confidence(self) -> float:
        return self.candidates[0][1]


def _window_offsets(
    ref: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    *,
    window_s: float,
    hop_s: float,
    feature: str,
    min_overlap_ratio: float,
    near_tie_ratio: float = NEAR_TIE_RATIO,
    offset_tolerance_s: float = 0.25,
    ref_envelope: "tuple[np.ndarray, float] | None" = None,
) -> "list[_WindowMeasurement]":
    """One :class:`_WindowMeasurement` per analysis window.

    The offset conversion is the load-bearing line. The correlation reports where the
    WINDOW begins on the reference; the window itself begins ``start_s`` into the clip;
    so the offset the window implies for the clip as a whole is ``lag - start_s``.
    Skipping that subtraction would make every window of a correctly-aligned clip report
    a different offset, and no two of them would ever be grouped.

    Each window reports its near-tied rivals rather than only its argmax, because on a
    reference that repeats verbatim the argmax is a coin flip (issue #30). Resolving the
    flip is :func:`_consensus_choice`'s job, and it cannot do it from an answer that has
    already thrown the alternatives away.

    ``offset_tolerance_s`` is not a filter here — it sets how far apart two candidates
    must be to count as different lags at all, so one peak's shoulders cannot appear on
    the ballot as several rivals.
    """
    n_win = max(1, int(round(window_s * sample_rate)))
    n_hop = max(1, int(round(hop_s * sample_rate)))
    starts = list(range(0, max(1, len(clip) - n_win + 1), n_hop))
    # Cover the tail: without this, up to `window_s` of a clip whose length is not a
    # whole number of hops is never looked at, and a span ending there is truncated.
    if starts and starts[-1] + n_win < len(clip):
        starts.append(max(0, len(clip) - n_win))

    # Computed once for the whole call rather than per window — 2.11x measured, and
    # the reference does not change between windows.
    ref_env = ref_envelope
    if feature == "envelope" and ref_env is None:
        ref_env = onset_envelope(ref, sample_rate)
    min_separation = max(1, int(round(offset_tolerance_s * sample_rate)))

    out: "list[_WindowMeasurement]" = []
    for s0 in starts:
        win = clip[s0 : s0 + n_win]
        if len(win) == 0:
            continue
        candidates = _feature_candidates(
            ref,
            win,
            sample_rate,
            feature=feature,
            min_overlap_ratio=min_overlap_ratio,
            near_tie_ratio=near_tie_ratio,
            min_separation=min_separation,
            ref_envelope=ref_env,
        )
        start_s = s0 / sample_rate
        offsets = tuple(
            (lag / sample_rate - start_s, coeff) for lag, coeff in candidates
        )
        out.append(
            _WindowMeasurement(
                clip_start_s=start_s,
                clip_end_s=(s0 + len(win)) / sample_rate,
                candidates=offsets,
                vote_offset_s=offsets[0][0],
            )
        )
    return out


def _consensus_choice(
    windows: "list[_WindowMeasurement]", *, offset_tolerance_s: float
) -> "list[_WindowMeasurement]":
    """Let the near-ties vote: a window departs from the crowd only on real evidence.

    The fix for issue #30. Each window's near-tied candidates are ballots for an offset;
    an offset's SUPPORT is the number of distinct windows that could be reading it; and
    each window then takes whichever of its own candidates has the most support, keeping
    its own best-scoring one to break a tie.

    Why this is the right shape rather than a smoothing pass: the disagreement between
    windows on repetitive material is not noise, it is the signal. A spurious peak lands
    at a DIFFERENT lag in each window (it comes from wherever that window's content
    happens to also fit), while the true offset is the one lag every window has in
    common. So the offset that repeats across windows is the true one almost by
    construction, and no window is ever moved to a lag its own correlation did not
    already rate as a near-tie — which is what leaves a genuine stop/restart alone. A
    restarted take has NO peak near the old offset, so it has nothing to vote for there
    and departs freely.

    **Support alone cannot settle an exactly tiling reference**, where every window
    offers a candidate at every repeat and all of them draw the same support. The tie is
    broken by CONTINUITY — stay with the previous window's offset — which is the honest
    move, because on an exact tiling those offsets are all equally true and the only
    thing left to prefer is the one that describes the clip as one take. That the answer
    was arbitrary is not swallowed: it shows up as a low
    :attr:`AlignedSpan.support`. Continuity is deliberately the SECOND key: support is
    evidence gathered from the whole clip, and a local prior must not overrule it.
    """
    if not windows:
        return []
    ballots = [np.array([off for off, _ in w.candidates]) for w in windows]
    hypotheses = np.unique(np.concatenate(ballots))
    support = np.zeros(hypotheses.size, dtype=int)
    for ballot in ballots:
        support += (
            np.abs(hypotheses[:, None] - ballot[None, :]) <= offset_tolerance_s
        ).any(axis=1)

    out: "list[_WindowMeasurement]" = []
    anchor: float | None = None
    for window, ballot in zip(windows, ballots):
        # `hypotheses` holds every candidate value exactly, so this is a lookup.
        own = support[np.searchsorted(hypotheses, ballot)]
        drift = np.zeros(ballot.size) if anchor is None else np.abs(ballot - anchor)
        # Keys are applied last-first: most support, then nearest the previous window,
        # then the window's own ranking (its best-scoring candidate first).
        best = int(np.lexsort((np.arange(ballot.size), drift, -own))[0])
        chosen = window.candidates[best]
        rest = tuple(c for i, c in enumerate(window.candidates) if i != best)
        out.append(replace(window, candidates=(chosen,) + rest))
        anchor = chosen[0]
    return out


def _spans_from_windows(
    measured: "list[_WindowMeasurement]",
    *,
    min_confidence: float,
    offset_tolerance_s: float,
    merge_gap_s: float,
) -> "list[AlignedSpan]":
    """Group per-window measurements into maximal runs of one stable offset.

    A run continues while a window is confident enough AND agrees with the run's offset
    so far, taken as the median of its members — the representative value, and the same
    statistic the span reports.

    An earlier draft also carried a *hysteresis* threshold, on the theory that a window
    dipping just below the opening bar would split a good span. Mutation testing showed
    that guard could not fail, and measuring why is what produced the merge pass below:
    a window degraded by silence does not dip, it collapses to zero — far under any keep
    threshold — so hysteresis was never what decided. Merging same-offset neighbours is.
    """
    runs: "list[list[_WindowMeasurement]]" = []
    current: "list[_WindowMeasurement]" = []
    for w in measured:
        offset, coeff = w.offset_s, w.confidence
        if current:
            ref_offset = float(np.median([m.offset_s for m in current]))
            if (
                coeff >= min_confidence
                and abs(offset - ref_offset) <= offset_tolerance_s
            ):
                current.append(w)
                continue
            runs.append(current)
            current = []
        if coeff >= min_confidence:
            current = [w]
    if current:
        runs.append(current)

    spans = [_span_from_run(run, offset_tolerance_s=offset_tolerance_s) for run in runs]
    return _merge_same_offset(
        _disjoin(spans), offset_tolerance_s=offset_tolerance_s, merge_gap_s=merge_gap_s
    )


def _span_from_run(
    run: "list[_WindowMeasurement]", *, offset_tolerance_s: float
) -> "AlignedSpan":
    """One run of agreeing windows → one :class:`AlignedSpan`, support included.

    Both the offset and the confidence are the MEDIAN over the run's windows — a span is
    only as trustworthy as its typical window, and the max would let one lucky window
    speak for all of them.

    ``support`` is counted on the windows' own candidates — their independent argmax
    (:attr:`_WindowMeasurement.vote_offset_s`) at full weight and the rest of their
    ballot at :data:`BALLOT_VOTE_WEIGHT` — not on the offsets consensus assigned them.
    Counting the assigned offsets would be circular — every window in a run agrees
    with its run by construction, so the number would be 1.0 always and would carry no
    information at all. A run too short to hold a disagreement
    (:data:`MIN_WINDOWS_FOR_SUPPORT`) reports ``None`` for the same reason: 1.0 there
    would be a unanimous vote of one.

    ``margin`` is read off the same ballots at the same time (:func:`_support_margin`),
    because the question support cannot answer — whether any OTHER offset is backed by
    as much of this run's evidence — is answerable from exactly the same windows.
    """
    offset = float(np.median([m.offset_s for m in run]))
    support = _support_fraction(run, offset, offset_tolerance_s=offset_tolerance_s)
    margin = _support_margin(run, offset, offset_tolerance_s=offset_tolerance_s)
    return AlignedSpan(
        clip_start_s=run[0].clip_start_s,
        clip_end_s=run[-1].clip_end_s,
        offset_s=offset,
        confidence=float(np.median([m.confidence for m in run])),
        support=support,
        margin=margin,
    )


def _clamp_to_reference(
    spans: "list[AlignedSpan]", reference_duration: float
) -> "list[AlignedSpan]":
    """Trim every span to the clip time that actually lands ON the reference.

    A window is admitted while it overlaps the reference by ``min_overlap_ratio`` of
    itself, so a span can run past a reference edge by up to
    ``(1 - min_overlap_ratio) * window_s`` — and the reported extent then describes
    reference time that does not exist. Measured on a 30 s reference:
    ``reference_span == (-10.0, 40.0)``, a 50 s span of a 30 s reference. That is not
    imprecise, it is impossible, and no ``window_s`` tolerance makes it fit.

    The failure it produces downstream is silent and points the wrong way. A caller
    slicing ``ref[int(-10.0 * sr):int(40.0 * sr)]`` gets numpy's negative-index
    resolution: the start wraps to reference time 20 s and the end clamps to 30 s, so
    the caller receives the reference's **tail** for a span whose true content is the
    reference's **entire** length. At the head edge the same slice returns zero samples.
    Neither raises.

    **Both timelines are trimmed, not just the projection.** ``AlignedSpan`` documents
    ``reference_time = clip_time + offset_s`` as holding inside the span; clamping
    ``reference_span`` alone would leave the clip edges wide and make the two views
    disagree. Because the offset is constant within a span the edges move together, so
    trimming is exact — it recovers ground truth on both axes.

    A span left with nothing is dropped: it aligned to no part of the reference, so it
    is not an aligned span. (Unlike ``ClipAlignment``, where a non-overlapping CLIP is
    kept with ``overlaps=False`` because a *source* must never leave the addressable set
    as a side effect of being measured — a span is a measurement, not a source.)

    **Why the clamp alone, and not also a per-window coverage penalty.** The tempting
    root-cause fix is to scale each window's confidence by the fraction of it that lies
    over the reference — because the correlation normalizes by the energy of its
    OVERLAP, so a half-hanging window scores like a whole match (measured: 0.9846
    against 0.9845 for the clean interior windows, i.e. marginally HIGHER). That
    reasoning is sound at the window level and it was implemented; it was then removed,
    because it changed no observable outcome. A span's confidence is the MEDIAN over its
    windows, which absorbs the single edge window — measured, an edge-overrun span
    reported 0.9847 against an interior span's 0.9845 with the penalty applied. The
    extent is fixed here, exactly; a contaminated window is kept out of a run by the
    offset-agreement rule; and a penalty that cannot change an output is decoration.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        lo = max(span.clip_start_s, -span.offset_s)
        hi = min(span.clip_end_s, reference_duration - span.offset_s)
        if hi - lo <= 0:
            continue
        out.append(replace(span, clip_start_s=lo, clip_end_s=hi))
    return out


def _merge_same_offset(
    spans: "list[AlignedSpan]", *, offset_tolerance_s: float, merge_gap_s: float
) -> "list[AlignedSpan]":
    """Rejoin neighbours that agree on the offset across a short unverified gap.

    Two spans carrying the same offset are one take with an unusable patch, not two
    takes: a stop and a restart cannot resume in sync, because whatever the reference
    kept doing while the camera was stopped changes the offset. So a same-offset pair is
    positive evidence of continuity, and reporting it as two spans invents a discontinuity.

    Measured on a continuous 50 s take with 12 s of hard silence at its middle: the
    windows inside the silence score exactly 0.0, the run breaks, and the result is two
    spans BOTH reporting offset 10.00 — a stop/restart that never happened.

    ``merge_gap_s`` is what keeps this honest. The gap is clip time nothing verified, and
    correlation cannot tell "quiet" from "different material"; past one window the two
    spans stay separate rather than the merge claiming a range it never measured.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        if out:
            prev = out[-1]
            gap = span.clip_start_s - prev.clip_end_s
            if (
                abs(span.offset_s - prev.offset_s) <= offset_tolerance_s
                and gap <= merge_gap_s
            ):
                out[-1] = AlignedSpan(
                    clip_start_s=prev.clip_start_s,
                    clip_end_s=span.clip_end_s,
                    # Duration-weighted, so a long span is not dragged by a short one.
                    offset_s=(
                        prev.offset_s * prev.duration_s
                        + span.offset_s * span.duration_s
                    )
                    / max(prev.duration_s + span.duration_s, 1e-9),
                    confidence=min(prev.confidence, span.confidence),
                    support=_merge_support(prev, span),
                    margin=_merge_margin(prev, span),
                )
                continue
        out.append(span)
    return out


def _merge_support(a: "AlignedSpan", b: "AlignedSpan") -> "float | None":
    """The support of two merged spans — duration-weighted, and ``None`` if either is.

    Weighted like the offset, so a long span is not dragged by a short one. The
    ``None`` rule is the load-bearing half: a span that never measured its support has
    no fraction to average in, and defaulting it to 1.0 (or dropping it and keeping the
    other side's) would let a stretch nothing corroborated inherit the vouching of the
    stretch beside it. Unmeasured plus measured is unmeasured.
    """
    return _merge_statistic(
        a.support, b.support, a_duration_s=a.duration_s, b_duration_s=b.duration_s
    )


def _merge_margin(a: "AlignedSpan", b: "AlignedSpan") -> "float | None":
    """The margin of two merged spans — the same rule :func:`_merge_support` uses.

    Deliberately not recomputed from the merged run: the two spans were measured on
    their own windows and the merge joins them across a gap nothing verified, so there
    is no combined ballot to read a runner-up off. Averaging what was measured is the
    honest move, and the ``None`` rule matters more here than for support — a span with
    no runner-up measured must not inherit its neighbour's separation.
    """
    return _merge_statistic(
        a.margin, b.margin, a_duration_s=a.duration_s, b_duration_s=b.duration_s
    )


def _merge_statistic(
    a_value: "float | None",
    b_value: "float | None",
    *,
    a_duration_s: float,
    b_duration_s: float,
) -> "float | None":
    """Duration-weighted mean of two per-span statistics, ``None`` if either is ``None``."""
    if a_value is None or b_value is None:
        return None
    total = max(a_duration_s + b_duration_s, 1e-9)
    return (a_value * a_duration_s + b_value * b_duration_s) / total


def _disjoin(spans: "list[AlignedSpan]") -> "list[AlignedSpan]":
    """Trim overlaps so the result partitions the aligned clip time.

    Windows overlap by ``window_s - hop_s``, so the window straddling a boundary belongs
    to both takes and gets assigned to whichever it correlated better with. Left alone
    that makes consecutive spans overlap — measured at 10 s on a 20 s/10 s grid — and a
    clip instant claimed by two different offsets is not a fact about anything.

    **The earlier span's END wins**, rather than splitting the difference. Its last
    window is a positive measurement: it correlated at that offset over its whole extent
    without straddling. The later span's first window is the contaminated one — it can
    score well while containing seconds of the previous take, because half a matching
    window is enough. Measured on a clip with a true boundary at 50 s: first-wins
    reproduces it exactly, while a midpoint split puts both edges 5 s out.
    """
    out: "list[AlignedSpan]" = []
    for span in spans:
        if out and span.clip_start_s < out[-1].clip_end_s:
            start = out[-1].clip_end_s
            if span.clip_end_s - start <= 0:
                continue  # wholly swallowed by its predecessor
            span = replace(span, clip_start_s=start)
        out.append(span)
    return out
