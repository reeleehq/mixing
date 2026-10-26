"""Cutaways over a base video: timed inserts, the base timeline and audio kept.

A documentary-style edit: a talking video runs unbroken while, at timed
windows, the picture cuts away to other clips or stills. ``cutaways(base,
inserts, overlay=...)`` builds **one** ffmpeg ``filter_complex`` (``trim``,
``setpts``, ``overlay``, ``adelay``, ``amix`` -- no moviepy, nothing beyond
filters every ffmpeg has) and runs it in a single pass:

- each :class:`Insert` is cut at ``src_in``, scaled to the base frame
  (``fit``), shifted to its destination start ``t0`` and overlaid on the base
  only while ``t0 <= t < t1`` (a hard cut; a window outlasting the insert
  falls back to the base);
- the base audio keeps playing at full level; an insert's own audio is mixed in
  only when it has an ``audio_gain`` (linear), with short fades at its edges;
  ``amix normalize=0`` so mixing does not halve the base level;
- an optional ``overlay`` image (a caption, a bug) is shown only while some
  insert is showing.

Plan, then execute: :func:`plan_cutaways` is pure (given the probed
:class:`MediaInfo` of the files) and returns a :class:`CutawayPlan` holding the
filter string and ffmpeg arguments; ``cutaways(..., dry_run=True)`` returns that
plan without running anything. Geometry vocabulary (``fit``) is ``looks``'
``FitMode``; a ``looks`` fragment (``looks.vf(...)``) can be given as the
grade of the base or of the inserts only.

>>> info = MediaInfo(width=1280, height=720, fps=25.0, duration=10.0, has_audio=True)
>>> plan = plan_cutaways("base.mp4", [Insert("b.mp4", 1.0, 2.0, 4.0)],
...                      base_info=info, insert_infos={"b.mp4": info})
>>> "overlay=eof_action=pass:enable='gte(t,2)*lt(t,4)'" in plan.filter_complex
True
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from mixing.egress import Output, write_egress
from mixing.util import ffmpeg_exe

#: ``fit`` values (``looks.geometry.FitMode``): ``fill`` scales to cover the base
#: frame and crops the excess, ``fit`` letterboxes, ``stretch`` ignores aspect.
FIT_MODES = ("fill", "fit", "stretch")
DEFAULT_FIT = "fill"
DEFAULT_AUDIO_FADE_IN_S = 0.15
DEFAULT_AUDIO_FADE_OUT_S = 0.25
#: Still images used as an insert are held for the window; no audio.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
#: Silent base track synthesised when the base has no audio but inserts do.
SILENCE_SAMPLE_RATE = 48000
#: Default encode of the single pass.
DEFAULT_FFMPEG_ARGS = (
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
    "-c:a", "aac", "-b:a", "192k",
)  # fmt: skip
#: Where the ``overlay`` image sits on the frame (ffmpeg ``overlay`` x, y).
DEFAULT_OVERLAY_XY = ("0", "0")


@dataclass(frozen=True)
class Insert:
    """One cutaway: ``src`` from ``src_in`` shown on the base over ``[t0, t1)``.

    ``audio_gain`` is the linear gain of the insert's own audio mixed under the
    base; ``None`` (default) drops it. A still image is held for the window.
    """

    src: str | Path
    src_in: float
    t0: float
    t1: float
    audio_gain: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.t0 < self.t1:
            raise ValueError(f"insert window must satisfy 0 <= t0 < t1: {self}")
        if self.src_in < 0:
            raise ValueError(f"src_in must be >= 0: {self}")
        if self.audio_gain is not None and self.audio_gain < 0:
            raise ValueError(f"audio_gain must be >= 0: {self}")

    @property
    def duration(self) -> float:
        return round(self.t1 - self.t0, 3)

    @property
    def is_image(self) -> bool:
        return Path(self.src).suffix.lower() in IMAGE_SUFFIXES


@dataclass(frozen=True)
class MediaInfo:
    """What the plan needs to know about a file (see :func:`probe_media`)."""

    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


@dataclass(frozen=True)
class CutawayPlan:
    """The filter graph and ffmpeg arguments of a cutaway render (not yet run)."""

    inputs: tuple[str, ...]  # ffmpeg input arguments (``-i`` groups), in order
    filter_complex: str
    maps: tuple[str, ...]  # output stream labels
    encode_args: tuple[str, ...] = field(default=DEFAULT_FFMPEG_ARGS)

    def command(self, output: str | Path, *, ffmpeg: Optional[str] = None) -> list[str]:
        """The full ffmpeg command writing ``output``."""
        args = [ffmpeg or ffmpeg_exe(), "-y", "-v", "error", *self.inputs]
        args += ["-filter_complex", self.filter_complex]
        for label in self.maps:
            args += ["-map", label]
        return [*args, *self.encode_args, str(output)]


# --- probing ------------------------------------------------------------------

_SIZE = re.compile(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"([\d.]+)\s*fps")
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_NO_DURATION = re.compile("(0)(0)(0)").match("000")


def parse_media_info(ffmpeg_stderr: str) -> MediaInfo:
    """:class:`MediaInfo` from the text ``ffmpeg -i FILE`` prints.

    >>> parse_media_info('''Duration: 00:00:10.50, start: 0.0
    ...   Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 25 fps
    ...   Stream #0:1: Audio: aac, 48000 Hz''')
    MediaInfo(width=1920, height=1080, fps=25.0, duration=10.5, has_audio=True)
    """
    size = _SIZE.search(ffmpeg_stderr)
    if not size:
        raise ValueError("no video stream found in ffmpeg output")
    fps = _FPS.search(ffmpeg_stderr)
    h, m, s = (_DURATION.search(ffmpeg_stderr) or _NO_DURATION).groups()
    return MediaInfo(
        width=int(size.group(1)),
        height=int(size.group(2)),
        fps=float(fps.group(1)) if fps else 0.0,
        duration=int(h) * 3600 + int(m) * 60 + float(s),
        has_audio=bool(re.search(r"Stream #.*Audio:", ffmpeg_stderr)),
    )


def probe_media(path: str | Path) -> MediaInfo:
    """Probe ``path`` with ``ffmpeg -i`` (no ffprobe needed)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    try:
        return parse_media_info(proc.stderr)
    except ValueError as e:
        raise ValueError(f"cannot probe {path}: {e}") from e


# --- planning (pure) ------------------------------------------------------------


def _fit_filter(fit: str, width: int, height: int) -> str:
    if fit == "fill":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    if fit == "fit":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    return f"scale={width}:{height}"


def _chain(*parts: Optional[str]) -> str:
    return ",".join(p for p in parts if p)


def _num(x: float) -> str:
    return f"{x:.6g}"


def _input_args(insert: Insert, fps: float) -> list[str]:
    if insert.is_image:
        return ["-loop", "1", "-framerate", _num(fps), "-t", _num(insert.duration),
                "-i", str(insert.src)]  # fmt: skip
    return ["-i", str(insert.src)]


def _insert_video(
    i: int, insert: Insert, *, width: int, height: int, fps: float, fit: str,
    grade: Optional[str],
) -> str:  # fmt: skip
    start = "" if insert.is_image else f"start={_num(insert.src_in)}:"
    return (
        f"[{i}:v]"
        + _chain(
            f"trim={start}duration={_num(insert.duration)}",
            "setpts=PTS-STARTPTS",
            f"fps={_num(fps)}",
            _fit_filter(fit, width, height),
            "setsar=1",
            grade,
            "format=yuv420p",
            f"setpts=PTS+{_num(insert.t0)}/TB",
        )
        + f"[v{i}]"
    )


def _insert_audio(
    i: int, insert: Insert, *, fade_in_s: float, fade_out_s: float
) -> str:
    d = insert.duration
    ms = int(round(insert.t0 * 1000))
    return (
        f"[{i}:a]"
        + _chain(
            f"atrim=start={_num(insert.src_in)}:duration={_num(d)}",
            "asetpts=PTS-STARTPTS",
            f"afade=t=in:d={_num(fade_in_s)}",
            f"afade=t=out:st={_num(max(d - fade_out_s, 0))}:d={_num(fade_out_s)}",
            f"volume={_num(insert.audio_gain)}",
            f"adelay={ms}|{ms}",
        )
        + f"[a{i}]"
    )


def _base_audio(info: MediaInfo) -> str:
    """The base audio as ``[a0]``: its own track, or silence as long as it."""
    if info.has_audio:
        return "[0:a]anull[a0]"
    return (
        f"anullsrc=r={SILENCE_SAMPLE_RATE}:cl=stereo,"
        f"atrim=duration={_num(info.duration)}[a0]"
    )


def plan_cutaways(
    base: str | Path,
    inserts: Sequence[Insert],
    *,
    base_info: MediaInfo,
    insert_infos: Mapping[str, MediaInfo],
    overlay: str | Path | None = None,
    overlay_xy: tuple[str, str] = DEFAULT_OVERLAY_XY,
    fit: str = DEFAULT_FIT,
    base_grade: Optional[str] = None,
    insert_grade: Optional[str] = None,
    audio_fade_in_s: float = DEFAULT_AUDIO_FADE_IN_S,
    audio_fade_out_s: float = DEFAULT_AUDIO_FADE_OUT_S,
    encode_args: Sequence[str] = DEFAULT_FFMPEG_ARGS,
) -> CutawayPlan:
    """The :class:`CutawayPlan` of ``inserts`` over ``base`` (see module docs).

    Pure: nothing is probed or run; ``base_info`` and ``insert_infos`` (keyed by
    ``str(insert.src)``) say which files have audio. ``base_grade`` and
    ``insert_grade`` are ffmpeg filter fragments (e.g. ``looks.vf(plan)``)
    applied to the base / to the inserts only.
    """
    if fit not in FIT_MODES:
        raise ValueError(f"fit must be one of {FIT_MODES}, got {fit!r}")
    if not inserts:
        raise ValueError("cutaways needs at least one insert")
    width, height, fps = base_info.width, base_info.height, base_info.fps or 25.0

    inputs = ["-i", str(base)]
    fc = [
        f"[0:v]"
        + _chain(
            f"fps={_num(fps)}", f"scale={width}:{height}", base_grade, "format=yuv420p"
        )
        + "[b0]"
    ]
    last, mixed = "b0", []
    for i, ins in enumerate(inserts, start=1):
        inputs += _input_args(ins, fps)
        fc.append(
            _insert_video(
                i, ins, width=width, height=height, fps=fps, fit=fit, grade=insert_grade
            )  # fmt: skip
        )
        fc.append(
            f"[{last}][v{i}]overlay=eof_action=pass:"
            f"enable='gte(t,{_num(ins.t0)})*lt(t,{_num(ins.t1)})'[b{i}]"
        )
        last = f"b{i}"
        has_audio = not ins.is_image and insert_infos[str(ins.src)].has_audio
        if ins.audio_gain is not None and has_audio:
            fc.append(
                _insert_audio(
                    i, ins, fade_in_s=audio_fade_in_s, fade_out_s=audio_fade_out_s
                )
            )
            mixed.append(f"[a{i}]")

    if overlay is not None:
        n_in = 1 + len(inserts)
        inputs += ["-loop", "1", "-i", str(overlay)]
        windows = "+".join(f"gte(t,{_num(i.t0)})*lt(t,{_num(i.t1)})" for i in inserts)
        fc.append(f"[{n_in}:v]format=rgba[cap]")
        fc.append(
            f"[{last}][cap]overlay={overlay_xy[0]}:{overlay_xy[1]}:shortest=1:"
            f"enable='{windows}'[vout]"
        )
        last = "vout"

    maps = [f"[{last}]"]
    if mixed:
        fc.append(_base_audio(base_info))
        fc.append(
            f"[a0]{''.join(mixed)}amix=inputs={1 + len(mixed)}:"
            "duration=first:normalize=0[aout]"
        )
        maps.append("[aout]")
    elif base_info.has_audio:
        maps.append("0:a")
    return CutawayPlan(
        inputs=tuple(inputs),
        filter_complex=";".join(fc),
        maps=tuple(maps),
        encode_args=tuple(encode_args),
    )


# --- execution ------------------------------------------------------------------


def cutaways(
    base: str | Path,
    inserts: Sequence[Insert],
    *,
    overlay: str | Path | None = None,
    overlay_xy: tuple[str, str] = DEFAULT_OVERLAY_XY,
    fit: str = DEFAULT_FIT,
    base_grade: Optional[str] = None,
    insert_grade: Optional[str] = None,
    audio_fade_in_s: float = DEFAULT_AUDIO_FADE_IN_S,
    audio_fade_out_s: float = DEFAULT_AUDIO_FADE_OUT_S,
    ffmpeg_args: Sequence[str] = DEFAULT_FFMPEG_ARGS,
    dry_run: bool = False,
    probe: Callable[[str | Path], MediaInfo] = probe_media,
    output: Output = None,
) -> "Path | CutawayPlan":
    """Cut away from ``base`` to ``inserts`` at their windows, in one ffmpeg pass.

    ``base`` runs unbroken (its video timeline and audio); each :class:`Insert`
    covers ``[t0, t1)`` with its clip (from ``src_in``) or still image, scaled
    by ``fit`` (``"fill"`` crops to cover, ``"fit"`` letterboxes, ``"stretch"``)
    and graded by ``insert_grade`` only. An insert with an ``audio_gain`` has its
    audio mixed under the base at that linear gain. ``overlay`` (an image, e.g.
    a caption) shows only while an insert does, at ``overlay_xy``. Inserts are
    drawn in order, so a later one wins where windows overlap.

    Args:
        base: The base video.
        inserts: The cutaways (see :class:`Insert`).
        overlay: Optional image shown during insert windows only.
        overlay_xy: ``overlay`` position, ffmpeg expressions (default top-left).
        fit: ``"fill"`` (default), ``"fit"`` or ``"stretch"``.
        base_grade, insert_grade: ffmpeg filter fragments for the base / inserts.
        audio_fade_in_s, audio_fade_out_s: fades on each mixed insert audio.
        ffmpeg_args: Encode arguments (default x264 CRF 18 + AAC).
        dry_run: Return the :class:`CutawayPlan` (``plan.command(out)`` is the
            ffmpeg command) without running anything.
        probe: How a file's :class:`MediaInfo` is obtained (seam for tests).
        output: Where to put the result: ``None`` (``<base>_cutaways.<ext>``
            beside the base), a file path, a directory, or a callable sink.

    Returns:
        The output path (or the sink's return value); the plan if ``dry_run``.
    """
    base = Path(base)
    infos = {str(i.src): probe(i.src) for i in inserts if not i.is_image}
    infos.update(
        {
            str(i.src): MediaInfo(0, 0, 0.0, i.duration, False)
            for i in inserts
            if i.is_image
        }
    )
    plan = plan_cutaways(
        base, inserts, base_info=probe(base), insert_infos=infos, overlay=overlay,
        overlay_xy=overlay_xy, fit=fit, base_grade=base_grade,
        insert_grade=insert_grade, audio_fade_in_s=audio_fade_in_s,
        audio_fade_out_s=audio_fade_out_s, encode_args=ffmpeg_args,
    )  # fmt: skip
    if dry_run:
        return plan

    def write(path: Path) -> None:
        proc = subprocess.run(plan.command(path), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-2000:]}")

    return write_egress(
        output,
        default_path=base.with_name(f"{base.stem}_cutaways{base.suffix}"),
        write=write,
    )
