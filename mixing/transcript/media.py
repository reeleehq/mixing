"""ffmpeg-backed audio extraction and timeline-cut application.

These functions run one ffmpeg each, through :func:`mixing.ffmpeg.run_ffmpeg`;
they have no Python dependencies beyond the stdlib. Both take ``ffmpeg=`` (the
executable; default :func:`mixing.util.ffmpeg_exe`, which finds a pip-bundled
build too: ``"ffmpeg"`` for the one on ``PATH``) and ``timeout=`` (seconds the
run may take; it is killed, with what it started, and
:class:`~mixing.ffmpeg.FfmpegTimeout` raised). A failed run raises
``RuntimeError`` with the tail of ffmpeg's message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from mixing.ffmpeg import run_ffmpeg
from mixing.util import ffmpeg_exe

PathLike = Union[str, Path]


def extract_audio(
    input_path: PathLike,
    output_path: PathLike,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    bitrate: str = "96k",
    overwrite: bool = True,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Path:
    """Extract a mono mp3 from ``input_path`` (suitable for STT).

    Returns the output path. ``ffmpeg`` and ``timeout`` as in the module docstring.
    """
    cmd = [
        ffmpeg or ffmpeg_exe(),
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    with run_ffmpeg(cmd, what=f"extract the audio of {input_path!s}", timeout=timeout):
        pass
    return Path(output_path)


def apply_keeps(
    input_path: PathLike,
    output_path: PathLike,
    keeps: Sequence[dict],
    *,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 20,
    preset: str = "medium",
    audio_bitrate: str = "160k",
    faststart: bool = True,
    overwrite: bool = True,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Path:
    """Re-encode ``input_path`` keeping only the listed time ranges.

    Args:
        input_path: Input audio/video file.
        output_path: Output file (extension picks the container).
        keeps: ``[{"start": float, "end": float}, ...]`` time ranges in seconds.
        video_codec / audio_codec / crf / preset / audio_bitrate: Re-encode
            knobs forwarded to ffmpeg.
        faststart: Add ``-movflags +faststart`` (mp4 web-streaming).
        overwrite: Pass ``-y`` to ffmpeg.
        ffmpeg, timeout: As in the module docstring.

    Returns:
        The output path.
    """
    if not keeps:
        raise ValueError("keeps must contain at least one range")
    expr = "+".join(f"between(t,{k['start']:.3f},{k['end']:.3f})" for k in keeps)
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{expr}',asetpts=N/SR/TB"
    cmd = [
        ffmpeg or ffmpeg_exe(),
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-af",
        af,
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
    ]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))
    with run_ffmpeg(cmd, what=f"cut {input_path!s} to its keeps", timeout=timeout):
        pass
    return Path(output_path)
