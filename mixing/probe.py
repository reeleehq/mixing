"""One light media probe: ``ffprobe`` first, ffmpeg's banner as the fallback.

What a file holds (its displayed size, exact frame rate, frame count, stream
start times, rotation, duration, sound format), read by subprocess so that
importing this module pulls no media library (no moviepy, opencv, numpy or
PIL). :mod:`mixing.edit` and :mod:`mixing.video.cutaways` both read their
sources through it.

>>> from mixing.probe import probe  # doctest: +SKIP
>>> info = probe("talk.mp4")  # doctest: +SKIP
>>> info.fps, info.frames, info.width, info.height  # doctest: +SKIP
(Fraction(30000, 1001), 900, 1920, 1080)

**Two methods, one result.** :func:`probe` returns a frozen :class:`MediaProbe`
whose ``method`` says how it was read:

- ``"ffprobe"`` (exact): the rate is ``r_frame_rate`` as a
  :class:`~fractions.Fraction` (``30000/1001`` stays exact), the frame count is
  the counted video packets (or the container's ``nb_frames`` header), every
  stream has its own ``start_time``, the rotation comes from the display
  matrix (side data) or the ``rotate`` tag, and ``r_frame_rate`` is checked
  against ``avg_frame_rate`` (and the packet count against the duration) to
  detect a variable rate.
- ``"banner"`` (approximate): when no ``ffprobe`` sits beside
  :func:`mixing.util.ffmpeg_exe` or on ``PATH`` (imageio-ffmpeg's build ships
  none), the text ``ffmpeg -i`` prints is parsed by :func:`parse_banner`. The
  rate is the printed decimal snapped by :func:`snap_rate`, the frame count is
  estimated from the **container** duration, both streams are taken to start
  with the file, and a variable rate cannot be detected.

``ffprobe=`` is the seam: the default finds one (:func:`find_ffprobe`), a path
uses that binary, ``None`` forces the banner. A file that does not exist raises
:class:`~mixing.errors.MediaNotFoundError`; one ffmpeg cannot read raises
:class:`~mixing.errors.ProbeError` (both are ``ValueError``s).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

from mixing.egress import PathLike
from mixing.errors import MediaNotFoundError, ProbeError
from mixing.util import ffmpeg_exe

__all__ = [
    "MediaProbe",
    "probe",
    "parse_banner",
    "find_ffprobe",
    "snap_rate",
    "AUTO_FFPROBE",
    "FFPROBE_METHOD",
    "BANNER_METHOD",
]

# --- constants -----------------------------------------------------------------

#: ``probe(ffprobe=AUTO_FFPROBE)`` (the default) looks for an ffprobe with
#: :func:`find_ffprobe`; a path names one; ``None`` forces the banner.
AUTO_FFPROBE = "auto"
#: :attr:`MediaProbe.method` of an exact (``ffprobe``) and an approximate
#: (``ffmpeg -i`` banner) reading.
FFPROBE_METHOD = "ffprobe"
BANNER_METHOD = "banner"
#: Name of the ffprobe executable, looked for beside ffmpeg and then on ``PATH``.
FFPROBE_NAME = "ffprobe"
#: Relative disagreement between a stream's ``r_frame_rate`` and its
#: ``avg_frame_rate`` above which the source is taken to be variable-rate.
VFR_TOLERANCE = 0.01
#: Frames by which a stream's packet count may differ from its duration times
#: its rate before it is taken to be variable-rate (container rounding).
VFR_FRAME_SLACK = 2
#: NTSC rates are ``k * 1000 / 1001``; a float rate this close to one (or to
#: an integer) is taken to mean it exactly (ffmpeg prints ``29.97``).
NTSC_DENOMINATOR = 1001
NTSC_NUMERATOR_UNIT = 1000
RATE_SNAP_TOLERANCE = 0.01
#: Largest denominator a float rate that is neither is approximated with.
MAX_RATE_DENOMINATOR = 1001
#: Channel counts of the layouts ffmpeg's banner names.
CHANNEL_LAYOUTS = {"mono": 1, "stereo": 2, "2.1": 3, "quad": 4, "5.0": 5, "5.1": 6}
#: Channels assumed for a banner layout not in :data:`CHANNEL_LAYOUTS`.
DEFAULT_BANNER_CHANNELS = 2
#: A display rotation this many degrees off a half turn swaps width and height.
QUARTER_TURN_DEGREES = 90
HALF_TURN_DEGREES = 180
#: Clock arithmetic of ``HH:MM:SS.fff`` (the banner's duration, mkv's tags).
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
#: Characters of a failing tool's stderr quoted in a :class:`ProbeError`.
ERROR_TAIL_CHARS = 500
#: What marks a location ffmpeg opens that is not a local file (``https://``…).
URL_MARKER = "://"


# --- the result -----------------------------------------------------------------


@dataclass(frozen=True)
class MediaProbe:
    """What :func:`probe` read about a media file.

    Video fields are ``None`` (and ``has_video`` false) for an audio-only file;
    sound fields are ``None`` (and ``has_audio`` false) for a silent one.

    Attributes:
        method: ``"ffprobe"`` (exact) or ``"banner"`` (approximate, see the
            module docstring).
        duration: the container's duration (s), ``None`` when not stated.
        width, height: the **displayed** size of the first video stream (a
            90/270-degree display rotation swaps the coded size; ffmpeg
            rotates on decode).
        rotation: the display rotation in degrees, as ffmpeg reports it.
        fps: the nominal rate (``r_frame_rate``, else ``avg_frame_rate``); from
            the banner, its printed rate snapped by :func:`snap_rate`.
        avg_fps: ``avg_frame_rate`` (``None`` from the banner, or when unset).
        frames: the frame count at ``fps``: the video packets when the rate is
            constant and they are known, else the video duration times ``fps``.
        packets: the video packet count, counted (``count_packets=True``) or
            the container's ``nb_frames`` header; ``None`` when unknown.
        video_start: where the first video frame sits after the file's start
            (what an input seek counts from).
        video_duration: the video stream's own length (s).
        audio_start: where the sound starts after the file's start.
        sample_rate, channels: of the first audio stream.
        variable_rate: the picture is not at a constant rate (rates disagree,
            or the packets disagree with the duration).
        has_audio: the file has an audio stream.
    """

    method: str
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    rotation: float = 0.0
    fps: Optional[Fraction] = None
    avg_fps: Optional[Fraction] = None
    frames: Optional[int] = None
    packets: Optional[int] = None
    video_start: float = 0.0
    video_duration: Optional[float] = None
    audio_start: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    variable_rate: bool = False
    has_audio: bool = False

    @property
    def has_video(self) -> bool:
        return self.width is not None

    @property
    def exact(self) -> bool:
        """Whether this came from ffprobe (exact) rather than the banner."""
        return self.method == FFPROBE_METHOD

    @property
    def coded_size(self) -> Optional[Tuple[int, int]]:
        """``(width, height)`` as stored, before the display rotation."""
        if not self.has_video:
            return None
        if _is_quarter_turn(self.rotation):
            return self.height, self.width
        return self.width, self.height

    @property
    def audio_offset(self) -> Optional[float]:
        """How far the sound starts after the first video frame (negative: before)."""
        if self.audio_start is None:
            return None
        return self.audio_start - self.video_start


# --- small parsers ----------------------------------------------------------------


def snap_rate(fps: float) -> Fraction:
    """A float frame rate as the exact rate it stands for.

    A rate near an integer or an NTSC rate (``k * 1000 / 1001``) is taken to
    mean it; anything else is approximated with a denominator of at most
    :data:`MAX_RATE_DENOMINATOR`.

    >>> snap_rate(29.97), snap_rate(25.0), snap_rate(23.976), snap_rate(12.5)
    (Fraction(30000, 1001), Fraction(25, 1), Fraction(24000, 1001), Fraction(25, 2))
    """
    if abs(fps - round(fps)) < RATE_SNAP_TOLERANCE:
        return Fraction(round(fps))
    k = round(fps * NTSC_DENOMINATOR / NTSC_NUMERATOR_UNIT)
    ntsc = Fraction(k * NTSC_NUMERATOR_UNIT, NTSC_DENOMINATOR)
    if abs(fps - float(ntsc)) < RATE_SNAP_TOLERANCE:
        return ntsc
    return Fraction(fps).limit_denominator(MAX_RATE_DENOMINATOR)


def _is_quarter_turn(degrees: float) -> bool:
    return abs(round(degrees)) % HALF_TURN_DEGREES == QUARTER_TURN_DEGREES


def _frames_in(seconds: float, fps: Fraction) -> int:
    return round(Fraction(seconds) * fps)


def _ratio(text: Optional[str]) -> Optional[Fraction]:
    if not text or text.endswith("/0"):
        return None
    value = Fraction(text)
    return value if value > 0 else None


def _seconds(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _count(text) -> Optional[int]:
    text = str(text or "")
    return int(text) if text.isdigit() else None


def _clock(text) -> Optional[float]:
    """Seconds of an ``HH:MM:SS.fff`` text (what mkv stores per stream)."""
    parts = str(text or "").split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = (_seconds(part) for part in parts)
    if None in (hours, minutes, seconds):
        return None
    return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds


def _rotation(stream: dict) -> float:
    for datum in stream.get("side_data_list", ()):
        if "rotation" in datum:
            return float(datum["rotation"])
    return float(stream.get("tags", {}).get("rotate", 0))


def _unreadable(media, detail: str) -> ProbeError:
    """The error for a file ffmpeg/ffprobe could not open: missing, or unreadable."""
    text = str(media)
    if URL_MARKER not in text and not Path(text).exists():
        return MediaNotFoundError(f"no such media file: {media}")
    tail = detail.strip()[-ERROR_TAIL_CHARS:]
    return ProbeError(f"cannot probe {media}: {tail or 'ffmpeg could not read it'}")


# --- ffprobe (exact) ----------------------------------------------------------------


def find_ffprobe() -> Optional[str]:
    """The ffprobe beside :func:`mixing.util.ffmpeg_exe`, else on ``PATH``, else ``None``."""
    sibling = Path(ffmpeg_exe()).with_name(FFPROBE_NAME)
    return str(sibling) if sibling.is_file() else shutil.which(FFPROBE_NAME)


def _probe_with_ffprobe(binary: str, media, *, count_packets: bool) -> MediaProbe:
    argv = [binary, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_packets:
        argv.append("-count_packets")
    proc = subprocess.run([*argv, str(media)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise _unreadable(media, proc.stderr)
    probed = json.loads(proc.stdout or "{}")
    streams, fmt = probed.get("streams", []), probed.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    origin = _seconds(fmt.get("start_time")) or 0.0
    audio_fields = {}
    if audio is not None:
        audio_fields = dict(
            audio_start=(_seconds(audio.get("start_time")) or origin) - origin,
            sample_rate=_count(audio.get("sample_rate")),
            channels=_count(audio.get("channels")),
        )
    common = dict(
        method=FFPROBE_METHOD,
        duration=_seconds(fmt.get("duration")),
        has_audio=audio is not None,
        **audio_fields,
    )
    if video is None:
        return MediaProbe(**common)
    return MediaProbe(**common, **_video_fields(video, fmt, origin, count_packets))


def _video_fields(video: dict, fmt: dict, origin: float, count_packets: bool) -> dict:
    nominal = _ratio(video.get("r_frame_rate"))
    average = _ratio(video.get("avg_frame_rate"))
    fps = nominal or average
    packets = _count(video.get("nb_read_packets" if count_packets else "nb_frames"))
    video_start = (_seconds(video.get("start_time")) or origin) - origin
    # a stream's own duration is its length; mkv's DURATION tag and the
    # format's duration run from the file's start, so the video's start is
    # not part of the video
    end = _clock(video.get("tags", {}).get("DURATION")) or _seconds(fmt.get("duration"))
    duration = _seconds(video.get("duration")) or (end and end - video_start)
    timed = _frames_in(duration, fps) if duration and fps else None
    # avg_frame_rate is barely filled by mkv: the packet count against the
    # duration is the other witness of a variable rate
    variable = bool(
        (nominal and average and abs(average / nominal - 1) > VFR_TOLERANCE)
        or (packets and timed and abs(packets - timed) > VFR_FRAME_SLACK)
    )
    rotation = _rotation(video)
    width, height = int(video["width"]), int(video["height"])
    if _is_quarter_turn(rotation):
        width, height = height, width
    return dict(
        width=width,
        height=height,
        rotation=rotation,
        fps=fps,
        avg_fps=average,
        frames=packets if packets and not variable else timed,
        packets=packets,
        video_start=video_start,
        video_duration=duration or None,
        variable_rate=variable,
    )


# --- the banner (approximate) -------------------------------------------------------

_BANNER_INPUT = re.compile(r"^Input #\d+", re.MULTILINE)
_BANNER_VIDEO = re.compile(r"Stream #.*Video:.*")
_BANNER_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")
_BANNER_FPS = re.compile(r"([\d.]+)\s*fps")
_BANNER_AUDIO = re.compile(r"Stream #.*Audio:.*")
_BANNER_SOUND = re.compile(r"(\d+) Hz, ([\w.]+)")
_BANNER_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_BANNER_ROTATION = re.compile(r"rotation of (-?[\d.]+) degrees")


def parse_banner(text: str) -> MediaProbe:
    """A :class:`MediaProbe` from the text ``ffmpeg -i FILE`` prints (approximate).

    The banner states the container duration, the first video stream's coded
    size and printed rate, the display rotation and the first audio stream's
    rate and layout. It has no per-stream start times (both streams are taken
    to start with the file), no packet count (``frames`` is the container
    duration times the rate) and no second rate to detect a variable one.

    >>> found = parse_banner('''Duration: 00:00:10.50, start: 0.0
    ...   Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 29.97 fps
    ...   Stream #0:1: Audio: aac, 48000 Hz, stereo, fltp''')
    >>> found.fps, found.frames, (found.width, found.height), found.channels
    (Fraction(30000, 1001), 315, (1920, 1080), 2)
    """
    duration = _BANNER_DURATION.search(text)
    seconds = None
    if duration:
        h, m, s = duration.groups()
        seconds = int(h) * SECONDS_PER_HOUR + int(m) * SECONDS_PER_MINUTE + float(s)
    fields = dict(method=BANNER_METHOD, duration=seconds)
    audio = _BANNER_AUDIO.search(text)
    if audio:
        sound = _BANNER_SOUND.search(audio.group(0))
        fields.update(has_audio=True, audio_start=0.0)
        if sound:
            fields.update(
                sample_rate=int(sound.group(1)),
                channels=CHANNEL_LAYOUTS.get(sound.group(2), DEFAULT_BANNER_CHANNELS),
            )
    line = _BANNER_VIDEO.search(text)
    size = _BANNER_SIZE.search(line.group(0)) if line else None
    if size:
        width, height = int(size.group(1)), int(size.group(2))
        rotation = _BANNER_ROTATION.search(text)
        degrees = float(rotation.group(1)) if rotation else 0.0
        if _is_quarter_turn(degrees):
            width, height = height, width
        rate = _BANNER_FPS.search(line.group(0))
        fps = snap_rate(float(rate.group(1))) if rate else None
        fields.update(
            width=width,
            height=height,
            rotation=degrees,
            fps=fps,
            frames=_frames_in(seconds, fps) if seconds and fps else None,
            video_duration=seconds,
        )
    return MediaProbe(**fields)


def _probe_with_banner(media) -> MediaProbe:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(media)], capture_output=True, text=True
    )
    if not _BANNER_INPUT.search(proc.stderr):
        raise _unreadable(media, proc.stderr)
    return parse_banner(proc.stderr)


# --- the probe ----------------------------------------------------------------------


def probe(
    media: PathLike,
    *,
    ffprobe: Optional[str] = AUTO_FFPROBE,
    count_packets: bool = True,
) -> MediaProbe:
    """Probe ``media`` with ffprobe when there is one, else with ffmpeg's banner.

    Args:
        media: the file (or any location ffmpeg opens).
        ffprobe: the ffprobe binary. :data:`AUTO_FFPROBE` (default) finds one
            with :func:`find_ffprobe`; ``None`` forces the approximate banner
            (:func:`parse_banner`).
        count_packets: have ffprobe count the video packets (reads, but does
            not decode, the whole file) for an exact frame count. ``False``
            uses the container's ``nb_frames`` header, which mkv lacks.

    Raises:
        MediaNotFoundError: ``media`` does not exist.
        ProbeError: ffmpeg/ffprobe cannot read it.
    """
    binary = find_ffprobe() if ffprobe == AUTO_FFPROBE else ffprobe
    if binary:
        return _probe_with_ffprobe(binary, media, count_packets=count_packets)
    return _probe_with_banner(media)
