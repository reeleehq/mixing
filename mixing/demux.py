"""A muxed file's sound as WAV: :func:`audio_track`.

For the answer a video-to-audio model returns as a video (MMAudio, ThinkSound,
HunyuanVideo-Foley): the first audio stream decoded to PCM WAV at its own rate
and channel count (no resample, no downmix; 16-bit by default, ``codec=`` for
float), unlike :func:`mixing.transcript.media.extract_audio` (16 kHz mono mp3,
for speech-to-text). One ffmpeg run through :func:`mixing.ffmpeg.run_ffmpeg`
(``timeout=`` kills it, and what it started). Pure stdlib, so importing it
pulls no media library.

>>> from mixing.demux import audio_track  # doctest: +SKIP
>>> wav = audio_track(answer_mp4_bytes)  # doctest: +SKIP
>>> wav[:4], wav[8:12]  # doctest: +SKIP
(b'RIFF', b'WAVE')
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from mixing.egress import Output, PathLike, deliver
from mixing.errors import MediaNotFoundError
from mixing.ffmpeg import check_timeout, run_ffmpeg
from mixing.util import ffmpeg_exe

__all__ = ["audio_track", "DEFAULT_AUDIO_TRACK_CODEC", "DEFAULT_AUDIO_TRACK_TIMEOUT"]

#: :func:`audio_track`'s WAV sample format: 16-bit PCM (``"pcm_f32le"`` keeps a
#: float decoder's samples, aac's say, unquantized, at twice the size).
DEFAULT_AUDIO_TRACK_CODEC = "pcm_s16le"
#: Seconds :func:`audio_track` lets ffmpeg run before killing it.
DEFAULT_AUDIO_TRACK_TIMEOUT = 120.0
#: The file name :func:`audio_track` gives bytes input's WAV in an ``output`` directory.
_BYTES_TRACK_NAME = "audio.wav"
_WAV_SUFFIX = ".wav"
#: A location ffmpeg opens that is not a local file: a URL scheme of two or more
#: characters (``https:``, ``data:``), so a Windows drive (``C:``) is not one.
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")

Media = Union[bytes, bytearray, memoryview, PathLike]


def _is_url(text: str) -> bool:
    return bool(_URL_SCHEME.match(text))


def _is_bytes(media: Media) -> bool:
    return isinstance(media, (bytes, bytearray, memoryview))


def _is_local_file(text: str) -> bool:
    try:
        return Path(text).is_file()
    except (OSError, ValueError):  # a name too long for the filesystem, a NUL
        return False


@contextmanager
def _local_media(media: Media):
    """A location ffmpeg can open for ``media``: bytes are written to a
    temporary file (seekable, which an mp4 with its index at the end needs); a
    local file is given by its absolute path (``take:2.mp4`` would otherwise
    read to ffmpeg as a ``take:`` protocol); a URL is yielded as is."""
    if not _is_bytes(media):
        text = os.fspath(media)
        if _is_local_file(text):
            yield os.path.abspath(text)
        elif _is_url(text):
            yield text
        else:
            raise MediaNotFoundError(f"no such media file: {media}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "media"
        path.write_bytes(bytes(media))
        yield str(path)


def _track_name(media: Media) -> str:
    """The WAV's name in an ``output`` directory: a local file's stem, else ``audio.wav``."""
    if _is_bytes(media) or not _is_local_file(os.fspath(media)):
        return _BYTES_TRACK_NAME
    return Path(os.fspath(media)).stem + _WAV_SUFFIX


def audio_track(
    media: Media,
    *,
    output: Output = None,
    codec: str = DEFAULT_AUDIO_TRACK_CODEC,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_AUDIO_TRACK_TIMEOUT,
):
    """The first audio stream of ``media`` as PCM WAV, at its own rate and channels.

    No resample and no downmix; the samples are quantized to ``codec`` (16-bit
    by default: a 24-bit or float source loses precision there, ``"pcm_f32le"``
    keeps it). A stream with no samples gives a header-only WAV. ffmpeg writes a file, not a pipe, so the WAV header carries the
    final size (a plain RIFF WAV, so a track is limited to 4 GB).

    Args:
        media: the container, as bytes or a path, or a URL ffmpeg opens.
        output: the ``output`` protocol (:mod:`mixing.egress`): ``None``
            (default) returns the WAV bytes; a file path writes them there, a
            directory writes ``<input stem>.wav`` (``audio.wav`` for bytes or a
            URL) in it, both returning the ``Path``; a callable gets the bytes.
        codec: the WAV sample format, an ffmpeg PCM encoder
            (:data:`DEFAULT_AUDIO_TRACK_CODEC`).
        ffmpeg: the executable (default :func:`mixing.util.ffmpeg_exe`).
        timeout: seconds ffmpeg may run before it is killed
            (:data:`DEFAULT_AUDIO_TRACK_TIMEOUT`; ``None`` for no bound).

    Raises:
        ValueError: ``timeout`` is not ``None`` or a number of seconds > 0
            (checked before anything runs).
        MediaNotFoundError: ``media`` is a path to no file.
        FfmpegTimeout: ffmpeg ran past ``timeout`` (a ``RuntimeError``).
        RuntimeError: ffmpeg cannot read it, it has no audio stream, or ffmpeg
            wrote no WAV.
    """
    timeout = check_timeout(timeout)
    exe = ffmpeg or ffmpeg_exe()
    what = "take the audio track out of " + (
        "the given bytes" if _is_bytes(media) else str(os.fspath(media))
    )
    with _local_media(media) as source, tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / _BYTES_TRACK_NAME
        cmd = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", source,
               "-map", "0:a:0", "-c:a", codec, str(wav)]  # fmt: skip
        with run_ffmpeg(cmd, what=what, timeout=timeout):
            pass
        data = wav.read_bytes() if wav.is_file() else b""
    if not data:
        raise RuntimeError(f"ffmpeg could not {what}: it exited 0 and wrote no WAV")
    return deliver(
        data,
        output,
        write=lambda b, path: Path(path).write_bytes(b),
        default_name=_track_name(media),
    )
