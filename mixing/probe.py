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
  the counted video packets (or the container's ``nb_frames`` header), the
  video starts at its ``start_time`` and the sound at its first **decoded**
  frame (an opus stream's first packet precedes it by its pre-skip), cover
  art (``attached_pic``) is not taken for the picture, the rotation comes from the display
  matrix (side data) or the ``rotate`` tag, and ``r_frame_rate`` is checked
  against ``avg_frame_rate`` (and the packet count against the duration) to
  detect a variable rate.
- ``"banner"`` (approximate): when no ``ffprobe`` sits beside
  :func:`mixing.util.ffmpeg_exe` or on ``PATH`` (imageio-ffmpeg's build ships
  none), the text ``ffmpeg -i`` prints is parsed by :func:`parse_banner`. The
  rates are the printed decimals (``tbr`` as ``fps``, ``fps`` as ``avg_fps``)
  snapped by :func:`snap_rate`, the frame count is estimated from the
  **container** duration, both streams are assumed to start with the file, and
  a variable rate is seen only when the two printed rates disagree.

``ffprobe=`` is the seam: the default finds one (:func:`find_ffprobe`), a path
uses that binary, ``None`` forces the banner. A file that does not exist raises
:class:`~mixing.errors.MediaNotFoundError`; one ffmpeg cannot read raises
:class:`~mixing.errors.ProbeError` (both are ``ValueError``s).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

from mixing.egress import PathLike
from mixing.errors import MediaNotFoundError, ProbeError
from mixing.ffmpeg import FfmpegTimeout, check_timeout, run_ffmpeg
from mixing.util import ffmpeg_exe

__all__ = [
    "MediaProbe",
    "MediaStream",
    "probe",
    "probe_duration",
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
#: :attr:`MediaStream.kind` of a stream ffprobe states no ``codec_type`` for.
UNKNOWN_STREAM_KIND = "unknown"
#: How the banner (and ffprobe's ``codec_name``) says a stream has no codec.
NO_CODEC = "none"
#: Name of the ffprobe executable, looked for beside ffmpeg and then on ``PATH``.
FFPROBE_NAME = "ffprobe"
#: ffprobe ``-read_intervals`` reading only the first packet (of the selected
#: stream): enough to see when its first decoded frame plays.
FIRST_PACKET_INTERVAL = "%+#1"
#: Executable suffix carried over from ffmpeg's name to ffprobe's (Windows).
WINDOWS_EXE_SUFFIX = ".exe"
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
#: The banner abbreviates large rates with a ``k`` suffix (``90k tbr``).
KILO = 1000
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
#: Seconds :func:`probe_duration` waits for ``ffmpeg -i`` before giving up (a
#: stalled network path or a damaged file would otherwise hang the caller).
DEFAULT_PROBE_DURATION_TIMEOUT = 30.0


# --- the result -----------------------------------------------------------------


@dataclass(frozen=True)
class MediaStream:
    """One stream of a media file, as :func:`probe` lists them in :attr:`MediaProbe.streams`.

    Attributes:
        kind: ``"video"``, ``"audio"``, ``"subtitle"``, ``"data"``, ``"attachment"``
            (ffprobe's ``codec_type``, the banner's stream label, lower-cased). Cover
            art is a ``"video"`` stream with ``attached_pic``.
        codec: the codec's short name (``"h264"``, ``"aac"``), ``None`` when not stated.
        attached_pic: the stream is a still picture attached to the file (an mp3's or
            m4a's cover art), not a picture track.
    """

    kind: str
    codec: Optional[str] = None
    attached_pic: bool = False


@dataclass(frozen=True)
class MediaProbe:
    """What :func:`probe` read about a media file.

    Video fields are ``None`` (and ``has_video`` false) for an audio-only file;
    sound fields are ``None`` (and ``has_audio`` false) for a silent one.

    Attributes:
        method: ``"ffprobe"`` (exact) or ``"banner"`` (approximate, see the
            module docstring).
        duration: the container's duration (s), ``None`` when not stated.
        width, height: the size of the first video stream's decoded frames
            after the display rotation (a 90/270-degree rotation swaps the
            coded size, as ffmpeg's decode does). The sample aspect ratio is
            not applied: an anamorphic stream reports its storage size.
        rotation: the display rotation in degrees, as ffmpeg reports it.
        fps: the nominal rate (``r_frame_rate``, else ``avg_frame_rate``); from
            the banner, its ``tbr`` (else its ``fps``) snapped by :func:`snap_rate`.
        avg_fps: ``avg_frame_rate``; from the banner, its ``fps``. ``None``
            when unset.
        frames: the frame count as the stream plays: the video packets when
            known, else the video duration times :attr:`playback_fps`. At a
            constant rate that is the count at ``fps``; for a ``variable_rate``
            stream it is the frames decoded one by one (a 24 fps film whose
            ``r_frame_rate`` reads 120 has its 192 frames, not 956), and
            :meth:`frames_at` gives the count resampled to a timeline rate.
        packets: the video packet count, counted (``count_packets=True``) or
            the container's ``nb_frames`` header; ``None`` when unknown.
        video_start: where the first video frame sits after the file's start
            (what an input seek counts from).
        video_duration: the video stream's own length (s).
        audio_start: where the sound's first decoded frame plays after the
            file's start (from the banner, assumed ``0.0``: it states no
            stream start times).
        sample_rate, channels: of the first audio stream.
        variable_rate: the picture is not at a constant rate (the nominal and
            average rates disagree by more than :data:`VFR_TOLERANCE`, or, from
            ffprobe, the packets disagree with the duration).
        has_audio: the file has an audio stream.
        streams: every stream of the file, in file order (:class:`MediaStream`).
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
    streams: Tuple[MediaStream, ...] = ()

    @property
    def has_video(self) -> bool:
        return self.width is not None

    @property
    def playback_fps(self) -> Optional[Fraction]:
        """The rate the frames actually arrive at, decoded one by one.

        ``fps`` (``r_frame_rate``) is the lowest rate that represents every
        timestamp, not the rate frames arrive at: a 24 fps film with sub-frame
        holes between its shots reads ``120/1`` (reeleehq/an#195, cutan's
        ``stream_fps``). So for a ``variable_rate`` stream the average rate is
        the playback rate; otherwise ``fps`` (``avg_fps`` when it is unset).

        >>> MediaProbe("ffprobe", fps=Fraction(120), avg_fps=Fraction(5760, 239),
        ...            variable_rate=True).playback_fps
        Fraction(5760, 239)
        >>> MediaProbe("ffprobe", fps=Fraction(24), avg_fps=Fraction(24)).playback_fps
        Fraction(24, 1)
        """
        return _playback_rate(self.fps, self.avg_fps, self.variable_rate)

    def frames_at(self, rate) -> Optional[int]:
        """The frame count of the video resampled by time to ``rate`` (fps):
        its duration times ``rate``; ``None`` when the duration is unknown.

        What a renderer with a ``rate`` timeline has from a variable-rate
        source (each timeline frame the nearest source frame), as opposed to
        :attr:`frames`, the frames the stream holds.

        >>> MediaProbe("ffprobe", video_duration=8.0).frames_at(120)
        960
        """
        if not self.video_duration:
            return None
        return _frames_in(self.video_duration, Fraction(rate))

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


def _frames_in_at(seconds: Optional[float], fps: Optional[Fraction]) -> Optional[int]:
    """:func:`_frames_in`, or ``None`` when either is unknown (or zero)."""
    return _frames_in(seconds, fps) if seconds and fps else None


def _playback_rate(
    fps: Optional[Fraction], average: Optional[Fraction], variable: bool
) -> Optional[Fraction]:
    """:attr:`MediaProbe.playback_fps` from the nominal and average rates."""
    if variable and average:
        return average
    return fps or average


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


def _start(stream: dict, origin: float) -> float:
    """``stream``'s ``start_time`` after the file's (``origin`` when unset; a
    stated ``0.0`` is a start, not a missing value)."""
    start = _seconds(stream.get("start_time"))
    return (origin if start is None else start) - origin


def _rates_disagree(nominal: Optional[Fraction], average: Optional[Fraction]) -> bool:
    """Whether a nominal and an average rate are far enough apart to mean VFR."""
    return bool(nominal and average and abs(average / nominal - 1) > VFR_TOLERANCE)


def _unreadable(media, proc: subprocess.CompletedProcess) -> ProbeError:
    """The error for a file ffmpeg/ffprobe could not open: missing, or unreadable."""
    text = str(media)
    if URL_MARKER not in text and not Path(text).exists():
        return MediaNotFoundError(f"no such media file: {media}")
    tool = Path(proc.args[0]).name
    tail = proc.stderr.strip()[-ERROR_TAIL_CHARS:]
    detail = f": {tail}" if tail else ""  # a crash (negative code) may print nothing
    return ProbeError(
        f"cannot probe {media}: {tool} could not read it "
        f"(exit code {proc.returncode}){detail}"
    )


def _run(
    cmd: list, *, what: str, deadline: Optional[float]
) -> subprocess.CompletedProcess:
    """``cmd`` run to completion through :func:`mixing.ffmpeg.run_ffmpeg` (its
    output decoded as text), within what is left before ``deadline`` (a
    monotonic time; ``None`` for no bound).

    Raises:
        FfmpegTimeout: the deadline passed (the run, and what it started, killed).
        OSError: ``cmd[0]`` cannot be run.
    """
    left = None if deadline is None else deadline - time.monotonic()
    if left is not None and left <= 0:
        raise FfmpegTimeout(f"no time left to {what}")
    with run_ffmpeg(
        cmd, what=what, stdout=subprocess.PIPE, timeout=left, check=False
    ) as proc:
        out = proc.stdout.read()
    return subprocess.CompletedProcess(
        cmd, proc.returncode, out.decode(errors="replace"), proc.message
    )


# --- ffprobe (exact) ----------------------------------------------------------------


def _first_audio_start(
    binary: str, media, audio: dict, origin: float, *, deadline: Optional[float]
) -> float:
    """When the first audio stream's first **decoded** frame plays, after the
    file's start (``origin``).

    Not its ``start_time``, which is its first packet: an opus decoder drops
    its pre-skip samples, so a stream whose first packet is at -0.007 s
    plays from 0.000 s (aac and mp3 do start at their first packet). Falls
    back to ``start_time`` (see :func:`_start`) when no frame is reported.
    """
    proc = _run(
        [binary, "-v", "error", "-select_streams", "a:0", "-show_frames",
         "-read_intervals", FIRST_PACKET_INTERVAL, "-show_entries", "frame=pts_time",
         "-of", "json", str(media)],
        what=f"read the first audio frame of {media!s}", deadline=deadline,
    )  # fmt: skip
    ok = not proc.returncode
    frames = json.loads(proc.stdout or "{}").get("frames", []) if ok else []
    decoded = next((_seconds(f.get("pts_time")) for f in frames), None)
    return _start(audio, origin) if decoded is None else decoded - origin


def _is_picture_stream(stream: dict) -> bool:
    """A video stream that moves (not an mp3's or m4a's cover art)."""
    attached = stream.get("disposition", {}).get("attached_pic", 0)
    return stream.get("codec_type") == "video" and not attached


def _has_dir(command: str) -> bool:
    """Whether ``command`` names a location (``./ffmpeg``, ``/usr/bin/ffmpeg``), not
    a bare name to look up on ``PATH``."""
    return any(sep and sep in command for sep in (os.sep, os.altsep))


def find_ffprobe(*, ffmpeg: Optional[str] = None) -> Optional[str]:
    """The ffprobe beside ``ffmpeg`` (default :func:`mixing.util.ffmpeg_exe`; a bare
    name is looked up on ``PATH``), else on ``PATH``, else ``None``."""
    named = ffmpeg_exe() if ffmpeg is None else ffmpeg
    if not _has_dir(named):  # a bare name: the one on PATH
        named = shutil.which(named) or named
    ffmpeg_path = Path(named)
    # ffprobe.exe beside ffmpeg.exe (Windows); any other suffix is not an
    # extension (imageio's Linux build is ``ffmpeg-linux-x86_64-v7.0.2``)
    is_exe = ffmpeg_path.suffix.lower() == WINDOWS_EXE_SUFFIX
    suffix = WINDOWS_EXE_SUFFIX if is_exe else ""
    sibling = ffmpeg_path.with_name(FFPROBE_NAME + suffix)
    if _has_dir(named) and sibling.is_file():
        return os.path.abspath(sibling)  # ``./ffprobe``'s Path reads as a bare name
    return shutil.which(FFPROBE_NAME)


def _probe_with_ffprobe(
    binary: str, media, *, count_packets: bool, deadline: Optional[float]
) -> MediaProbe:
    argv = [binary, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_packets:
        argv.append("-count_packets")
    try:
        proc = _run([*argv, str(media)], what=f"probe {media!s}", deadline=deadline)
    except FfmpegTimeout:  # a TimeoutError, so an OSError: not a missing binary
        raise
    except OSError as e:  # not MediaNotFoundError: it is the binary that is missing
        raise ProbeError(f"cannot run ffprobe {binary!r}: {e}") from e
    if proc.returncode != 0:
        raise _unreadable(media, proc)
    probed = json.loads(proc.stdout or "{}")
    streams, fmt = probed.get("streams", []), probed.get("format", {})
    video = next((s for s in streams if _is_picture_stream(s)), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    origin = _seconds(fmt.get("start_time")) or 0.0
    audio_fields = {}
    if audio is not None:
        audio_fields = dict(
            audio_start=_first_audio_start(
                binary, media, audio, origin, deadline=deadline
            ),
            sample_rate=_count(audio.get("sample_rate")),
            channels=_count(audio.get("channels")),
        )
    common = dict(
        method=FFPROBE_METHOD,
        duration=_seconds(fmt.get("duration")),
        has_audio=audio is not None,
        streams=tuple(_ffprobe_stream(stream) for stream in streams),
        **audio_fields,
    )
    if video is None:
        return MediaProbe(**common)
    return MediaProbe(**common, **_video_fields(video, fmt, origin, count_packets))


def _ffprobe_stream(stream: dict) -> MediaStream:
    return MediaStream(
        kind=str(stream.get("codec_type", UNKNOWN_STREAM_KIND)),
        codec=_codec(stream.get("codec_name")),
        attached_pic=bool(stream.get("disposition", {}).get("attached_pic", 0)),
    )


def _codec(name: Optional[str]) -> Optional[str]:
    """A stated codec name, ``None`` for none (``"none"``, or not stated)."""
    return None if name in (None, "", NO_CODEC) else name


def _video_fields(video: dict, fmt: dict, origin: float, count_packets: bool) -> dict:
    nominal = _ratio(video.get("r_frame_rate"))
    average = _ratio(video.get("avg_frame_rate"))
    fps = nominal or average
    packets = _count(video.get("nb_read_packets" if count_packets else "nb_frames"))
    video_start = _start(video, origin)
    # a stream's own duration is its length. Otherwise: mkv's DURATION tag is
    # the stream's end timestamp (on the same clock as start_time), and the
    # format's duration runs from the file's start (origin): either way the
    # video's own start is not part of the video
    tag_end = _clock(video.get("tags", {}).get("DURATION"))
    end = tag_end - origin if tag_end else _seconds(fmt.get("duration"))
    duration = _seconds(video.get("duration")) or (end and end - video_start)
    timed = _frames_in_at(duration, fps)
    # avg_frame_rate is barely filled by mkv: the packet count against the
    # duration is the other witness of a variable rate
    variable = bool(
        _rates_disagree(nominal, average)
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
        frames=packets
        or _frames_in_at(duration, _playback_rate(fps, average, variable)),
        packets=packets,
        video_start=video_start,
        video_duration=duration or None,
        variable_rate=variable,
    )


# --- the banner (approximate) -------------------------------------------------------

_BANNER_INPUT = re.compile(r"^Input #\d+", re.MULTILINE)
#: A stream's line in the banner: ``Stream #0:1[0x2](und): Video: h264 (High), ...``
#: (kind, then codec).
_BANNER_STREAM = re.compile(
    r"^\s*Stream #(\d+):(\d+).*?: (\w+): (\w+)?(.*)$", re.MULTILINE
)
_BANNER_VIDEO = re.compile(r"Stream #.*Video:.*")
#: How the banner marks a video stream that is cover art, not a picture track.
BANNER_ATTACHED_PIC = "(attached pic)"
_BANNER_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})")
_BANNER_FPS = re.compile(r"([\d.]+)(k?)\s*fps")
_BANNER_TBR = re.compile(r"([\d.]+)(k?)\s*tbr")
_BANNER_AUDIO = re.compile(r"Stream #.*Audio:.*")
_BANNER_SOUND = re.compile(r"(\d+) Hz, ([\w.]+)")
_BANNER_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_BANNER_ROTATION = re.compile(r"rotation of (-?[\d.]+) degrees")


def _banner_seconds(text: str) -> Optional[float]:
    """The container duration the banner states (``Duration: HH:MM:SS.ff``), or ``None``."""
    found = _BANNER_DURATION.search(text)
    if found is None:
        return None
    h, m, s = found.groups()
    return int(h) * SECONDS_PER_HOUR + int(m) * SECONDS_PER_MINUTE + float(s)


def _banner_rate(pattern: re.Pattern, line: str) -> Optional[Fraction]:
    """A rate the banner prints (``29.97 fps``, ``30 tbr``, ``90k tbr``), snapped."""
    found = pattern.search(line)
    if not found:
        return None
    value = float(found.group(1)) * (KILO if found.group(2) else 1)
    rate = snap_rate(value)
    return rate if rate > 0 else None  # ``0.0033 fps`` snaps to 0: no rate


def parse_banner(text: str) -> MediaProbe:
    """A :class:`MediaProbe` from the text ``ffmpeg -i FILE`` prints (approximate).

    The banner states the container duration, the first video stream's coded
    size, its two rates (``N tbr`` is ffmpeg's guess at ``r_frame_rate`` and
    becomes ``fps``; ``N fps`` is ``avg_frame_rate`` and becomes ``avg_fps``;
    both snapped by :func:`snap_rate`), the display rotation and the first
    audio stream's rate and layout. It has no per-stream start times (both
    streams are **assumed** to start with the file: ``video_start`` and
    ``audio_start`` are ``0.0``) and no packet count (``frames`` is the
    container duration times ``fps``), so a variable rate is detected only
    when the two printed rates disagree.

    >>> found = parse_banner('''Duration: 00:00:10.50, start: 0.0
    ...   Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 29.97 fps
    ...   Stream #0:1: Audio: aac, 48000 Hz, stereo, fltp''')
    >>> found.fps, found.frames, (found.width, found.height), found.channels
    (Fraction(30000, 1001), 315, (1920, 1080), 2)
    """
    seconds = _banner_seconds(text)
    # a stream in two programs (mpegts), or in a stream group (ffmpeg 8+), is
    # printed more than once: one record per (input, index), in index order
    by_index = {}
    for m in _BANNER_STREAM.finditer(text):
        by_index.setdefault(
            (int(m.group(1)), int(m.group(2))),
            MediaStream(
                kind=m.group(3).lower(),
                codec=_codec(m.group(4)),
                attached_pic=BANNER_ATTACHED_PIC in m.group(0),
            ),
        )
    streams = tuple(by_index[key] for key in sorted(by_index))
    fields = dict(method=BANNER_METHOD, duration=seconds, streams=streams)
    audio = _BANNER_AUDIO.search(text)
    if audio:
        sound = _BANNER_SOUND.search(audio.group(0))
        fields.update(has_audio=True, audio_start=0.0)
        if sound:
            fields.update(
                sample_rate=int(sound.group(1)),
                channels=CHANNEL_LAYOUTS.get(sound.group(2), DEFAULT_BANNER_CHANNELS),
            )
    line = next(
        (
            m
            for m in _BANNER_VIDEO.finditer(text)
            if BANNER_ATTACHED_PIC not in m.group(0)
        ),
        None,
    )
    size = _BANNER_SIZE.search(line.group(0)) if line else None
    if size:
        width, height = int(size.group(1)), int(size.group(2))
        rotation = _BANNER_ROTATION.search(text)
        degrees = float(rotation.group(1)) if rotation else 0.0
        if _is_quarter_turn(degrees):
            width, height = height, width
        average = _banner_rate(_BANNER_FPS, line.group(0))
        fps = _banner_rate(_BANNER_TBR, line.group(0)) or average
        variable = _rates_disagree(fps, average)
        fields.update(
            width=width,
            height=height,
            rotation=degrees,
            fps=fps,
            avg_fps=average,
            frames=_frames_in_at(seconds, _playback_rate(fps, average, variable)),
            video_duration=seconds,
            variable_rate=variable,
        )
    return MediaProbe(**fields)


def _probe_with_banner(
    media, *, ffmpeg: Optional[str], deadline: Optional[float]
) -> MediaProbe:
    proc = _run(  # a missing ffmpeg is an OSError: not the file's fault
        [ffmpeg or ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(media)],
        what=f"read the banner of {media!s}",
        deadline=deadline,
    )
    if not _BANNER_INPUT.search(proc.stderr):
        raise _unreadable(media, proc)
    return parse_banner(proc.stderr)


# --- the probe ----------------------------------------------------------------------


def probe_duration(
    media: PathLike,
    *,
    timeout: Optional[float] = DEFAULT_PROBE_DURATION_TIMEOUT,
    ffmpeg: Optional[str] = None,
) -> Optional[float]:
    """The container duration of ``media`` in seconds, from ``ffmpeg -i``, or ``None``.

    The cheap question alone: :func:`probe`'s banner reading (one ``ffmpeg -i``,
    no ffprobe lookup), and ``None`` rather than an error whenever the duration
    cannot be had: the file is missing or unreadable, its container states no
    duration, or ffmpeg is still running ``timeout`` seconds after it started.
    Unlike :func:`probe` and :mod:`mixing.decode`, whose ``timeout`` defaults to
    no bound and raises :class:`~mixing.ffmpeg.FfmpegTimeout`, this one is
    bounded by default and answers ``None``: it is the question a caller asks
    before deciding anything, and must not hang it. On expiry ffmpeg is killed
    with whatever it started (see :func:`mixing.ffmpeg.run_ffmpeg`).

    Args:
        media: the file (or any location ffmpeg opens).
        timeout: seconds to wait for ffmpeg (default
            :data:`DEFAULT_PROBE_DURATION_TIMEOUT`); ``None`` (or ``inf``)
            waits forever.
        ffmpeg: the executable, as :func:`probe` takes it.

    Raises:
        ValueError: ``timeout`` is not ``None`` or a number of seconds > 0
            (see :func:`mixing.ffmpeg.check_timeout`).
        RuntimeError: no ffmpeg executable can be found (see
            :func:`mixing.util.ffmpeg_exe`).
        OSError: the ``ffmpeg`` given cannot be run.
    """
    try:
        return probe(media, ffprobe=None, ffmpeg=ffmpeg, timeout=timeout).duration
    except (ProbeError, FfmpegTimeout):
        return None


def probe(
    media: PathLike,
    *,
    ffprobe: Optional[str] = AUTO_FFPROBE,
    count_packets: bool = True,
    ffmpeg: Optional[str] = None,
    timeout: Optional[float] = None,
) -> MediaProbe:
    """Probe ``media`` with ffprobe when there is one, else with ffmpeg's banner.

    Args:
        media: the file (or any location ffmpeg opens).
        ffprobe: the ffprobe binary. :data:`AUTO_FFPROBE` (default) finds one
            with :func:`find_ffprobe` (beside ``ffmpeg``); ``None`` forces the
            approximate banner (:func:`parse_banner`).
        count_packets: have ffprobe count the video packets (reads, but does
            not decode, the whole file) for an exact frame count. ``False``
            uses the container's ``nb_frames`` header, which mkv lacks.
        ffmpeg: the ffmpeg whose banner is read, and beside which ffprobe is
            looked for (default :func:`mixing.util.ffmpeg_exe`; ``"ffmpeg"`` for
            the one on ``PATH``), as :mod:`mixing.decode` takes it.
        timeout: seconds the whole probe may take (every ffprobe/ffmpeg run it
            makes); the run still going then is killed, with what it started,
            and :class:`~mixing.ffmpeg.FfmpegTimeout` is raised. ``None``
            (default) for no bound.

    Raises:
        MediaNotFoundError: ``media`` does not exist (a ``FileNotFoundError``
            too, so tell it from a missing ``ffmpeg`` by this class).
        ProbeError: ffmpeg/ffprobe cannot read it, or the named ``ffprobe``
            cannot be run.
        OSError: the ``ffmpeg`` read for the banner cannot be run (a missing
            binary is a ``FileNotFoundError``), as with :mod:`mixing.decode`.
        FfmpegTimeout: the probe ran past ``timeout`` (a ``TimeoutError``).
        ValueError: ``ffprobe=""``, or ``timeout`` is not ``None`` or seconds > 0.
    """
    if ffprobe == "":
        raise ValueError(
            "ffprobe='' names no binary: pass a path, AUTO_FFPROBE to find one, "
            "or None for the banner fallback"
        )
    budget = check_timeout(timeout)
    deadline = None if budget is None else time.monotonic() + budget
    binary = find_ffprobe(ffmpeg=ffmpeg) if ffprobe == AUTO_FFPROBE else ffprobe
    try:
        if binary is not None:
            return _probe_with_ffprobe(
                binary, media, count_packets=count_packets, deadline=deadline
            )
        return _probe_with_banner(media, ffmpeg=ffmpeg, deadline=deadline)
    except FfmpegTimeout as e:  # name the caller's budget, not what was left of it
        raise FfmpegTimeout(
            f"probing {media} took more than {budget:g} s; it was killed"
        ) from e
