"""Video utilities for subtitle embedding and video processing.

This module provides utilities for embedding subtitles into videos with two approaches:
1. Fast FFmpeg-based approach (recommended for production)
2. MoviePy CompositeVideoClip approach (for compatibility)
"""

from typing import Optional, Callable
from pathlib import Path
import logging
import os
import re
import subprocess
from dataclasses import dataclass

import numpy as np
import moviepy as mp

from config2py import process_path

from ..egress import Output, write_egress

# Canonical SRT time + cue helpers live in mixing.srt; re-exported here for
# backward compatibility (``to_srt_time`` is part of the video public surface).
from mixing.srt import (
    srt_time_to_seconds,
    seconds_to_srt_time,
    to_srt_time,
    shift_srt_timestamps,
)

logger = logging.getLogger(__name__)


def _find_audio_peaks(
    video_path: str | Path,
    *,
    first_n_seconds_to_sample: float = 20.0,
    window_size: float = 0.1,
    min_peak_threshold: float = 0.001,
) -> list[dict]:
    """
    Analyze audio waveform to find peaks indicating sound activity.

    Args:
        video_path: Path to video file
        first_n_seconds_to_sample: Duration to analyze (default 20 seconds)
        window_size: Size of analysis window in seconds (default 0.1s)
        min_peak_threshold: Minimum threshold for peak detection (0-1 scale, default 0.001)

    Returns:
        List of dicts with keys: 'time' (seconds), 'intensity' (std dev),
        'variation' (change from previous window)
    """
    video = mp.VideoFileClip(str(video_path))

    try:
        if video.audio is None:
            return []

        # Limit to sampling duration
        duration = min(first_n_seconds_to_sample, video.duration)

        # Extract audio data
        fps = video.audio.fps
        audio_array = video.audio.to_soundarray(fps=fps)

        # Handle stereo by averaging channels
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Calculate window size in samples
        samples_per_window = int(window_size * fps)
        num_windows = int(duration / window_size)

        peaks = []
        prev_intensity = 0.0

        for i in range(num_windows):
            start_sample = i * samples_per_window
            end_sample = min(start_sample + samples_per_window, len(audio_array))

            if end_sample <= start_sample:
                break

            window_data = audio_array[start_sample:end_sample]

            # Calculate intensity (standard deviation of waveform)
            intensity = float(np.std(window_data))

            # Calculate variation (change from previous window)
            variation = intensity - prev_intensity

            time_offset = i * window_size

            # Only record if above threshold
            if intensity >= min_peak_threshold:
                peaks.append(
                    {
                        "time": time_offset,
                        "intensity": intensity,
                        "variation": variation,
                    }
                )

            prev_intensity = intensity

        return peaks

    finally:
        video.close()


def find_audio_start_offset(
    video_path: str | Path,
    *,
    first_n_seconds_to_sample: float = 20.0,
    intensity_threshold: float = 0.001,
    variation_multiplier: float = 2.0,
) -> float:
    """
    Determine when speaking/audio actually starts in a video.

    Analyzes audio peaks and finds the first significant audio activity,
    typically indicating when speech begins.

    Args:
        video_path: Path to video file
        first_n_seconds_to_sample: Duration to analyze (default 20 seconds)
        intensity_threshold: Minimum intensity to consider as "audio start" (default 0.001)
        variation_multiplier: Multiplier for variation-based detection

    Returns:
        Time offset in seconds where audio starts (0.0 if no audio detected)

    Note:
        The default threshold of 0.001 is calibrated to detect speech start reliably
        across various audio levels. Adjust if needed for very quiet or loud audio.
    """
    peaks = _find_audio_peaks(
        video_path, first_n_seconds_to_sample=first_n_seconds_to_sample
    )

    if not peaks:
        return 0.0

    # Calculate statistics
    intensities = [p["intensity"] for p in peaks]
    variations = [p["variation"] for p in peaks]

    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)
    mean_variation = np.mean(variations)
    std_variation = np.std(variations)

    # Find first significant peak
    # Look for intensity above threshold AND significant variation
    for peak in peaks:
        # Strong intensity signal
        intensity_significant = peak["intensity"] >= intensity_threshold

        # Or significant jump in variation (sudden increase)
        variation_significant = (
            peak["variation"] > mean_variation + variation_multiplier * std_variation
        )

        if intensity_significant or variation_significant:
            return peak["time"]

    # Fallback: return time of first peak
    return peaks[0]["time"] if peaks else 0.0


def auto_shift_srt_to_start(
    srt_content: str,
    *,
    video_path: str | Path | None = None,
    auto_detect_audio_start: bool = False,
    start_time: float | bool | Callable[[str | Path], float] | None = None,
) -> str:
    """
    Automatically shift SRT timestamps to align with a target start time.

    By default, keeps subtitles at their original timestamps. Can optionally align
    to a specific start time, or auto-detect when audio actually begins.

    Args:
        srt_content: SRT file content as string
        video_path: Path to video file (required if auto_detect_audio_start=True or start_time=True)
        auto_detect_audio_start: If True, detect actual audio start time (default False)
        start_time: Target start time for first subtitle, or True for auto-detect, or callable.
            - None (default): No change, keep original subtitle timestamps
            - float: Shift subtitles so first one starts at this time (in seconds)
            - True: Auto-detect audio start and align first subtitle to that time
            - callable: Use this function to find the target start time

    Returns:
        Modified SRT content with shifted timestamps

    Examples:
        >>> srt = '''1
        ... 00:43:12,187 --> 00:43:13,817
        ... First subtitle
        ...
        ... 2
        ... 00:43:20,557 --> 00:43:22,087
        ... Second subtitle'''
        >>> shifted = auto_shift_srt_to_start(srt, start_time=0.0)
        >>> '00:00:00,000 --> 00:00:01,630' in shifted
        True
        >>> '00:00:08,370 --> 00:00:09,900' in shifted
        True

    Note:
        The shift amount is calculated as: shift = start_time - current_first_subtitle_time
        Example: If subtitles start at 10s and start_time=3s, shift will be -7s,
        moving all timestamps back by 7 seconds so first subtitle starts at 3s.
    """
    # Find first subtitle timestamp
    lines = srt_content.split("\n")
    first_subtitle_time = None

    for line in lines:
        match = re.match(r"(\d+:\d+:\d+,\d+)\s+-->", line)
        if match:
            first_timestamp = match.group(1)
            first_subtitle_time = srt_time_to_seconds(first_timestamp)
            break

    if first_subtitle_time is None:
        # No timestamps found, return unchanged
        return srt_content

    # If no start_time specified and not auto-detecting, keep original timestamps
    if start_time is None and not auto_detect_audio_start:
        return srt_content

    # Determine the target start time
    should_detect_audio = auto_detect_audio_start or start_time is True

    if should_detect_audio:
        if video_path is None:
            raise ValueError(
                "video_path is required when auto_detect_audio_start=True or start_time=True"
            )

        # Use custom function if provided, otherwise use default
        if callable(start_time):
            target_start_time = start_time(video_path)
        else:
            target_start_time = find_audio_start_offset(video_path)

        # Calculate shift: if subtitles start at 10s and target is 3s, shift by -7s
        shift_amount = target_start_time - first_subtitle_time

        logger.info("Audio detected at: %.2fs", target_start_time)
        logger.info("Subtitles currently start at: %.2fs", first_subtitle_time)
        logger.info(
            "Shifting by: %.2fs (first subtitle -> %.2fs)",
            shift_amount,
            target_start_time,
        )
    elif isinstance(start_time, (int, float)):
        # User provided explicit target start time
        target_start_time = start_time
        shift_amount = target_start_time - first_subtitle_time

        logger.info("Subtitles currently start at: %.2fs", first_subtitle_time)
        logger.info(
            "Shifting by: %.2fs (first subtitle -> %.2fs)",
            shift_amount,
            target_start_time,
        )
    else:
        # Should not reach here, but fallback to no change
        return srt_content

    return shift_srt_timestamps(srt_content, shift_amount)


def fix_srt_file(
    input_srt: str | Path,
    output_srt: str | Path | None = None,
    *,
    video_path: str | Path | None = None,
    auto_detect_audio_start: bool = False,
    start_time: float | bool | Callable[[str | Path], float] | None = None,
) -> Path:
    """
    Fix an SRT file by shifting timestamps to align with a target start time.

    By default, keeps subtitles at their original timestamps. Can optionally align
    to a specific start time, or auto-detect when audio actually begins.

    Args:
        input_srt: Path to input SRT file
        output_srt: Path to output SRT file (default: add suffix based on operation)
        video_path: Path to video file (for audio-based start time detection)
        auto_detect_audio_start: If True, align with detected audio start (default False)
        start_time: Target start time for first subtitle, or True for auto-detect, or callable.
            - None (default): No change, keep original subtitle timestamps
            - float: Shift subtitles so first one starts at this time (in seconds)
            - True: Auto-detect audio start and align first subtitle to that time
            - callable: Use this function to find the target start time

    Returns:
        Path to the fixed SRT file

    Examples:
        Basic usage (no change): fix_srt_file('subtitles.srt')
        Shift to start at 0: fix_srt_file('subtitles.srt', start_time=0.0)
        With audio detection: fix_srt_file('subtitles.srt', video_path='video.mp4',
                                          auto_detect_audio_start=True)
        Explicit start time: fix_srt_file('subtitles.srt', start_time=3.5)
    """
    input_path = Path(input_srt)

    if output_srt is None:
        has_adjustment = auto_detect_audio_start or start_time is not None
        suffix = "_audio_aligned" if has_adjustment else "_fixed"
        output_path = input_path.with_stem(input_path.stem + suffix)
    else:
        output_path = Path(output_srt)

    # Read original content
    srt_content = input_path.read_text()

    # Shift timestamps
    fixed_content = auto_shift_srt_to_start(
        srt_content,
        video_path=video_path,
        auto_detect_audio_start=auto_detect_audio_start,
        start_time=start_time,
    )

    # Write fixed content
    output_path.write_text(fixed_content)

    logger.info("Fixed SRT file saved to: %s", output_path)
    logger.info("Original first subtitle: %s", _get_first_timestamp(srt_content))
    logger.info("Fixed first subtitle: %s", _get_first_timestamp(fixed_content))

    return output_path


def _get_first_timestamp(srt_content: str) -> str:
    """Get the first timestamp from SRT content."""
    for line in srt_content.split("\n"):
        match = re.match(r"(\d+:\d+:\d+,\d+)\s+-->", line)
        if match:
            return match.group(1)
    return "Not found"


@dataclass
class SubtitleStyle:
    """Configuration for subtitle appearance."""

    font_size: int = 24
    color: str = "white"
    position: tuple[str, str] = ("center", "bottom")
    font_name: str = "Arial"

    def to_text_clip_kwargs(self) -> dict:
        """Convert style to TextClip kwargs."""
        return {
            "font_size": self.font_size,
            "color": self.color,
        }

    def to_ffmpeg_style(self) -> str:
        """Convert to FFmpeg subtitle style string."""
        alignment_map = {
            ("center", "bottom"): 2,
            ("center", "top"): 8,
            ("center", "center"): 5,
            ("left", "bottom"): 1,
            ("right", "bottom"): 3,
        }
        alignment = alignment_map.get(self.position, 2)

        color_map = {
            "white": "FFFFFF",
            "yellow": "FFFF00",
            "black": "000000",
            "red": "FF0000",
            "green": "00FF00",
            "blue": "0000FF",
        }
        color_hex = color_map.get(self.color.lower(), "FFFFFF")

        return f"FontSize={self.font_size},FontName={self.font_name},PrimaryColour=&H{color_hex}&,Alignment={alignment}"


def _parse_srt_timestamp(timestamp: str) -> str:
    """
    Parse SRT timestamp and convert to moviepy format.

    >>> _parse_srt_timestamp('00:00:01,500')
    '00:00:01.500'
    """
    return timestamp.replace(",", ".")


def _parse_srt_subtitle(subtitle_block: str) -> Optional[tuple[str, str, str]]:
    """
    Parse a single SRT subtitle block.

    Returns (start_time, end_time, text) or None if invalid.

    >>> block = '''1
    ... 00:00:01,000 --> 00:00:03,000
    ... Hello, world!'''
    >>> _parse_srt_subtitle(block)
    ('00:00:01.000', '00:00:03.000', 'Hello, world!')
    """
    lines = subtitle_block.strip().split("\n")
    if len(lines) < 3:
        return None

    try:
        time_info = lines[1].split(" --> ")
        if len(time_info) != 2:
            return None
        start_time = _parse_srt_timestamp(time_info[0])
        end_time = _parse_srt_timestamp(time_info[1])
        text = " ".join(lines[2:])
        return start_time, end_time, text
    except (IndexError, ValueError):
        return None


def generate_subtitle_clips(
    subtitles: str,
    video_clip: mp.VideoClip,
    *,
    style: Optional[SubtitleStyle] = None,
    **text_clip_kwargs,
) -> list[mp.TextClip]:
    """Generate subtitle clips from SRT content."""
    if style is None:
        style = SubtitleStyle()

    text_kwargs = {**style.to_text_clip_kwargs(), **text_clip_kwargs}

    subtitle_clips = []
    for subtitle_block in subtitles.split("\n\n"):
        parsed = _parse_srt_subtitle(subtitle_block)
        if parsed is None:
            continue

        start_time, end_time, text = parsed

        try:
            txt_clip = mp.TextClip(
                text=text, size=(video_clip.w, None), method="caption", **text_kwargs
            )
            txt_clip = (
                txt_clip.with_start(start_time)
                .with_end(end_time)
                .with_position(style.position)
            )
            subtitle_clips.append(txt_clip)
        except Exception as e:
            logger.warning("Failed to create subtitle clip: %s", e)
            continue

    return subtitle_clips


Filepath = str


def write_subtitles_in_video(
    video: Filepath,
    subtitles: str | None = None,
    output: Output = None,
    *,
    embed_subtitles: bool = True,
    style: Optional[SubtitleStyle] = None,
    use_ffmpeg: bool = True,
    auto_detect_audio_start: bool = False,
    start_time: float | bool | Callable[[str | Path], float] | None = None,
    **subtitle_kwargs,
) -> Path:
    """
    Write subtitles in a video, preserving audio and video quality.

    Uses FFmpeg by default for 100x speedup over MoviePy's CompositeVideoClip.

    Args:
        video: Path to input video file
        subtitles: Path to SRT file, SRT content string, or None (auto-detect)
        output: Where to put the result — None (save beside the input,
            auto-named), a file path, a directory (auto-named), or a callable
            sink. See mixing.egress.
        embed_subtitles: Whether to embed subtitles (default True)
        style: SubtitleStyle configuration (optional)
        use_ffmpeg: Use FFmpeg directly for speed (default True, recommended)
        auto_detect_audio_start: If True, align subtitles with detected audio start (default False)
        start_time: Target start time for first subtitle, or True for auto-detect, or callable.
            - None (default): No change, keep original subtitle timestamps
            - float: Shift subtitles so first one starts at this time (in seconds)
            - True: Auto-detect audio start and align first subtitle to that time
            - callable: Use this function to find the target start time
        **subtitle_kwargs: Additional kwargs for subtitle styling (MoviePy only)

    Returns:
        Path to the output video file
    """
    video_path = Path(process_path(video))

    if not embed_subtitles:
        default_path = _get_output_path(None, video_path, suffix="_copy")
        import shutil

        return write_egress(
            output,
            default_path=default_path,
            write=lambda out_path: shutil.copy2(video_path, out_path),
        )

    srt_content = _read_srt_content(subtitles, video_path)
    default_path = _get_output_path(None, video_path)

    # Apply timestamp adjustment if requested
    if auto_detect_audio_start or start_time is not None:
        srt_content = auto_shift_srt_to_start(
            srt_content,
            video_path=video_path,
            auto_detect_audio_start=auto_detect_audio_start,
            start_time=start_time,
        )

    def _write(out_path: Path) -> None:
        if use_ffmpeg:
            _embed_subtitles_ffmpeg(video_path, srt_content, out_path, style)
        else:
            _embed_subtitles_moviepy(
                video_path, srt_content, out_path, style, **subtitle_kwargs
            )

    return write_egress(output, default_path=default_path, write=_write)


def _embed_subtitles_ffmpeg(
    video_path: Path,
    srt_content: str,
    output_path: Path,
    style: Optional[SubtitleStyle] = None,
) -> Path:
    """Embed subtitles using FFmpeg directly (FAST - recommended)."""
    import tempfile

    if style is None:
        style = SubtitleStyle()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        temp_srt_path = f.name
        f.write(srt_content)

    try:
        style_str = style.to_ffmpeg_style()

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles='{temp_srt_path}':force_style='{style_str}'",
            "-c:a",
            "copy",
            "-y",
            str(output_path),
        ]

        logger.info("Running FFmpeg to embed subtitles...")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("FFmpeg stderr: %s", result.stderr)
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

        logger.info("Subtitles embedded successfully")
        return output_path

    finally:
        try:
            os.unlink(temp_srt_path)
        except:
            pass


def _embed_subtitles_moviepy(
    video_path: Path,
    srt_content: str,
    output_path: Path,
    style: Optional[SubtitleStyle] = None,
    **subtitle_kwargs,
) -> Path:
    """Embed subtitles using MoviePy CompositeVideoClip (SLOW - legacy method)."""
    logger.warning(
        "Using slow MoviePy method. Consider use_ffmpeg=True for 100x speedup."
    )

    video_clip = mp.VideoFileClip(str(video_path))

    subtitle_clips = generate_subtitle_clips(
        srt_content, video_clip, style=style, **subtitle_kwargs
    )

    video_with_subtitles = mp.CompositeVideoClip([video_clip] + subtitle_clips)
    video_with_subtitles = video_with_subtitles.with_audio(video_clip.audio)

    video_with_subtitles.write_videofile(
        str(output_path), codec="libx264", audio_codec="aac", fps=video_clip.fps
    )

    video_clip.close()

    return output_path


def _read_srt_content(subtitles: str | None, video_path: Path) -> str:
    """Resolve ``subtitles`` (path, SRT content, or None) to SRT text.

    ``None`` looks for a sibling ``.srt`` file next to ``video_path``.
    """
    if subtitles is None:
        subtitles_path = video_path.with_suffix(".srt")
        if not subtitles_path.exists():
            raise FileNotFoundError(
                f"No subtitle file found at {subtitles_path}. "
                f"Please provide subtitles explicitly."
            )
        return subtitles_path.read_text()
    if os.path.isfile(subtitles):
        subtitles_path = process_path(subtitles)
        return Path(subtitles_path).read_text()
    return subtitles


def _get_output_path(
    output_video: str | None, video_path: Path, suffix: str = "_with_subtitles"
) -> Path:
    """Generate output video path."""
    if output_video is None:
        output_video_path = video_path.with_stem(video_path.stem + suffix)
        output_video_path = output_video_path.with_suffix(video_path.suffix)
    else:
        output_video_path = Path(output_video)
    return output_video_path
