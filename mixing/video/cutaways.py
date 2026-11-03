"""Cutaways over a base video: timed inserts, the base timeline and audio kept.

A documentary-style edit: a talking video runs unbroken while, at timed
windows, the picture cuts away to other clips or stills. ``cutaways(base,
items, overlay=...)`` builds **one** ffmpeg ``filter_complex`` (``setpts``,
``overlay``, ``adelay``, ``amix`` -- no moviepy, nothing beyond filters every
ffmpeg has) and runs it in a single pass:

- each :class:`Cutaway` is seeked to ``src_in`` at the input, scaled to the base
  frame (``fit``, default ``"fill"``: cover and crop), shifted to its start
  ``t0`` and overlaid on the base only while ``t0 <= t < t1`` (a hard cut, or,
  with ``transition_in`` (a ``looks.Transition``), an ``xfade`` from what is
  on screen into the item over ``[t0, t0 + duration_s)``; a ``looks.Ripple``
  enters by its dissolve and adds its warp to the finished picture; a window
  outlasting the clip falls back to the base);
- the base audio keeps playing at full level; a cutaway's own audio is mixed in
  only when it has an ``audio_gain`` (linear), with short fades at its edges;
  ``amix normalize=0`` so mixing does not halve the base level;
- an optional ``overlay`` image (a caption, a bug) is shown only while some
  cutaway is showing; an item's own ``overlay`` replaces it during that item;
- an optional ``post`` filter fragment runs on the finished picture (letterbox,
  grain: ``looks.vf(...)``), so the whole edit stays one pass.

Plan, then execute: :func:`plan_cutaways` returns a :class:`CutawayPlan` (the
filter string and ffmpeg arguments) after probing the files, and
:func:`render_cutaways` runs it; :func:`cutaways` is the two in one call. Fit
geometry is ``looks``' ``placement`` (its ``FitMode`` vocabulary), and a
``looks`` fragment (``looks.vf(...)``) can grade the base or the cutaways only.

>>> info = MediaInfo(width=1280, height=720, fps=25.0, duration=10.0, has_audio=True)
>>> plan = plan_cutaways("base.mp4", [Cutaway("b.mp4", 1.0, 2.0, 4.0)],
...                      probe=lambda path: info)
>>> "overlay=eof_action=pass:enable='gte(t,2)*lt(t,4)'" in plan.filter_complex
True
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from looks import Effect, Ripple, Transition
from looks.geometry import FitMode, Size, ffmpeg_chain, placement

from mixing.egress import Output, write_egress
from mixing.probe import MediaProbe, parse_banner, probe
from mixing.util import ffmpeg_exe

#: ``fit`` is a ``looks.geometry.FitMode``: ``fill`` scales to cover the base
#: frame and crops the excess, ``fit`` letterboxes, ``stretch`` ignores aspect.
DEFAULT_FIT: FitMode = "fill"
#: Frame rate used for stills and probes that report none.
DEFAULT_FPS = 25.0
#: A clip may fall this much (s) short of its window without a warning.
SOURCE_OVERRUN_TOLERANCE_S = 0.1
DEFAULT_AUDIO_FADE_IN_S = 0.15
DEFAULT_AUDIO_FADE_OUT_S = 0.25
#: Still images used as an insert are held for the window; no audio.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
#: Silent base track synthesised when the base has no audio but inserts do.
SILENCE_SAMPLE_RATE = 48000
#: Channel layout every mixed cutaway audio is brought to (the silent base is too).
MIX_CHANNEL_LAYOUT = "stereo"
#: Default encode of the single pass.
DEFAULT_FFMPEG_ARGS = (
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
    "-c:a", "aac", "-b:a", "192k",
)  # fmt: skip
#: Where the ``overlay`` image sits on the frame (ffmpeg ``overlay`` x, y).
DEFAULT_OVERLAY_XY = ("0", "0")

#: How an item enters: a cross-fade (``looks.Transition``), or a ``looks.Ripple``
#: (its dissolve, plus its warp on the finished picture).
Entrance = Union[Transition, Ripple]


@dataclass(frozen=True)
class Cutaway:
    """One cutaway: ``src`` from ``src_in`` shown on the base over ``[t0, t1)``.

    ``audio_gain`` is the linear gain of the insert's own audio mixed under the
    base; ``None`` (default) drops it. A still image is held for the window.

    ``transition_in`` (a ``looks.Transition``) makes the item *enter*: it is
    revealed over ``[t0, t0 + duration_s)`` on top of whatever shows under it
    then, playing from ``src_in`` at ``t0`` as with a cut. Under it is the base,
    or an earlier cutaway whose window still covers the entrance (to go from
    one cutaway into the next, end the first at ``t0 + duration_s``, not at
    ``t0``). A transition centred on a cut at ``c`` (``lead`` before it,
    ``tail`` after) is ``t0=c - lead``, ``src_in - lead``, ``duration_s=lead + tail``. ``overlay`` is an image shown during
    this item's window instead of :func:`plan_cutaways`' ``overlay`` (a big
    caption on one window, the small one elsewhere), at ``overlay_xy`` (default:
    the plan's).

    ``transition_in`` may be a ``looks.Ripple``: the item enters by its
    ``dissolve``, and :func:`plan_cutaways` runs its warp
    (``effect(offset=t0)``) on the finished picture, the only stream that holds
    both shots (the entrance's own ``xfade`` holds the incoming item alone).
    """

    src: str | Path
    src_in: float
    t0: float
    t1: float
    audio_gain: Optional[float] = None
    transition_in: Optional[Entrance] = None
    overlay: str | Path | None = None
    overlay_xy: Optional[tuple[str, str]] = None

    def __post_init__(self):
        if not 0 <= self.t0 < self.t1:
            raise ValueError(f"insert window must satisfy 0 <= t0 < t1: {self}")
        if self.src_in < 0:
            raise ValueError(f"src_in must be >= 0: {self}")
        if self.audio_gain is not None and self.audio_gain < 0:
            raise ValueError(f"audio_gain must be >= 0: {self}")
        if self.entrance is not None and not (
            self.entrance.duration_s < self.t1 - self.t0
        ):
            raise ValueError(
                f"transition_in ({self.entrance.duration_s}s) must be shorter "
                f"than the window ({self.t1 - self.t0:g}s): {self}"
            )

    @property
    def duration(self) -> float:
        return round(self.t1 - self.t0, 3)

    @property
    def entrance(self) -> Optional[Transition]:
        """The cross-fade the item enters by (a ``Ripple``'s ``dissolve``)."""
        t = self.transition_in
        return t.dissolve if isinstance(t, Ripple) else t

    @property
    def warp(self) -> Optional[Effect]:
        """The ``looks`` effect this item adds to the finished picture, if any."""
        t = self.transition_in
        return t.effect(offset=self.t0) if isinstance(t, Ripple) else None

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


def _media_info(found: MediaProbe, *, no_video: str) -> MediaInfo:
    """The plan's view of a :class:`~mixing.probe.MediaProbe` (``no_video``: the
    ``ValueError`` message when it has no picture)."""
    if not found.has_video:
        raise ValueError(no_video)
    return MediaInfo(
        width=found.width,
        height=found.height,
        fps=float(found.fps) if found.fps else 0.0,
        duration=found.duration or 0.0,
        has_audio=found.has_audio,
    )


def parse_media_info(ffmpeg_stderr: str) -> MediaInfo:
    """:class:`MediaInfo` from the text ``ffmpeg -i FILE`` prints.

    The size is the **displayed** one: ffmpeg rotates a 90/270-degree display
    matrix on decode, so width and height are swapped for it. The parsing is
    :func:`mixing.probe.parse_banner`'s: ``fps`` is the banner's ``tbr`` (the
    nominal rate; its ``fps`` figure when there is no ``tbr``), snapped to the
    exact rate it stands for (``29.97`` reads as ``30000/1001``).

    >>> parse_media_info('''Duration: 00:00:10.50, start: 0.0
    ...   Stream #0:0: Video: h264, yuv420p, 1920x1080 [SAR 1:1], 25 fps
    ...   Stream #0:1: Audio: aac, 48000 Hz''')
    MediaInfo(width=1920, height=1080, fps=25.0, duration=10.5, has_audio=True)
    """
    found = parse_banner(ffmpeg_stderr)
    return _media_info(found, no_video="no video stream found in ffmpeg output")


def probe_media(path: str | Path) -> MediaInfo:
    """Probe ``path`` (a video or an image) with :func:`mixing.probe.probe`.

    ffprobe when there is one (exact rate), else ``ffmpeg -i``'s banner; the
    packets are not counted (the plan only needs the container duration).

    ``fps`` is the **nominal** rate (ffprobe's ``r_frame_rate``, the banner's
    ``tbr``), not the average one the banner's ``fps`` figure states (and
    that this probe returned before :mod:`mixing.probe`): for a
    variable-rate file they differ (an mp4 with frames dropped reads 30, not
    its average 15.52).
    """
    found = probe(path, count_packets=False)  # MediaNotFoundError / ProbeError
    return _media_info(found, no_video=f"cannot probe {path}: no video stream found")


# --- planning (pure) ------------------------------------------------------------


def _fit_filter(fit: FitMode, source: MediaInfo, base: MediaInfo) -> str:
    """``scale,crop/pad`` taking ``source`` to the base frame (``looks`` geometry)."""
    return ffmpeg_chain(
        placement(
            Size(source.width, source.height), Size(base.width, base.height), mode=fit
        )
    )


def _chain(*parts: Optional[str]) -> str:
    return ",".join(p for p in parts if p)


def _num(x: float) -> str:
    return f"{x:.6g}"


def _input_args(item: Cutaway, fps: float) -> list[str]:
    """ffmpeg input group for ``item``: seeked and limited at the input (fast)."""
    if item.is_image:
        return ["-loop", "1", "-framerate", _num(fps), "-t", _num(item.duration),
                "-i", str(item.src)]  # fmt: skip
    return ["-ss", _num(item.src_in), "-t", _num(item.duration), "-i", str(item.src)]


def _item_video(
    i: int, item: Cutaway, *, base: MediaInfo, source: MediaInfo, fps: float,
    fit: FitMode, grade: Optional[str],
) -> str:  # fmt: skip
    return (
        f"[{i}:v]"
        + _chain(
            "setpts=PTS-STARTPTS",
            f"fps={_num(fps)}",
            _fit_filter(fit, source, base),
            "setsar=1",
            grade,
            "format=yuv420p",
            f"setpts=PTS+{_num(item.t0)}/TB",
        )
        + f"[v{i}]"
    )


def _item_audio(i: int, item: Cutaway, *, fade_in_s: float, fade_out_s: float) -> str:
    d = item.duration
    ms = int(round(item.t0 * 1000))
    return (
        f"[{i}:a]"
        + _chain(
            "asetpts=PTS-STARTPTS",
            # downmix ourselves: amix's automatic one is ~2.6x too loud for 5.1
            f"aresample=ochl={MIX_CHANNEL_LAYOUT}:rematrix_maxval=1",
            f"afade=t=in:d={_num(fade_in_s)}",
            f"afade=t=out:st={_num(max(d - fade_out_s, 0))}:d={_num(fade_out_s)}",
            f"volume={_num(item.audio_gain)}",
            f"adelay={ms}:all=1",  # every channel, not just the first two
        )
        + f"[a{i}]"
    )


def _base_audio(info: MediaInfo) -> str:
    """The base audio as ``[a0]``: its own track, or silence as long as it."""
    if info.has_audio:
        return "[0:a]anull[a0]"
    return (
        f"anullsrc=r={SILENCE_SAMPLE_RATE}:cl={MIX_CHANNEL_LAYOUT},"
        f"atrim=duration={_num(info.duration)}[a0]"
    )


def _entrance(i: int, item: Cutaway, *, fps: float) -> tuple[list[str], str]:
    """``(filters, item label)`` making item ``i`` enter by its ``transition_in``.

    The item is split; its first ``duration_s`` is ``xfade``d from a fully
    transparent copy of itself (``offset=0``: the blend is the whole span) and
    put back at ``t0`` as ``[x{i}]``, to be overlaid on whatever is on screen
    then. Only the opacity rises: from a transparent *black* picture the colour
    would rise too, and a fade would dip ~26 % darker at its middle.
    Blending from transparency rather than from the main picture keeps the main
    stream unsplit: a split of it would hold every frame up to ``t0`` in memory
    (gigabytes for a late cutaway on a long base). Curves that only reveal the
    item (fades, dissolve, wipes, circles) are exact; curves that also move or
    darken the outgoing picture (slides, ``fadeblack``) move only the incoming one.
    """
    t0, d, rate = _num(item.t0), _num(item.entrance.duration_s), _num(fps)
    # xfade needs a stated constant rate on both inputs
    head = (
        f"trim=start={t0}:duration={d},setpts=PTS-STARTPTS,fps={rate},format=yuva420p"
    )
    return (
        [
            f"[v{i}]split[w{i}][wx{i}]",
            f"[wx{i}]{head},split[wf{i}][wz{i}]",
            f"[wz{i}]lutyuv=a=0[z{i}]",  # the same picture, fully transparent
            f"[z{i}][wf{i}]xfade=transition={item.entrance.curve}:"
            f"duration={d}:offset=0,setpts=PTS+{t0}/TB[x{i}]",
        ],
        f"w{i}",
    )


def _window(item: Cutaway, *, after_entrance: bool = False) -> str:
    if after_entrance and item.entrance is not None:
        start = _num(item.t0 + item.entrance.duration_s)
        return f"gte(t,{start})*lt(t,{_num(item.t1)})"
    return f"gte(t,{_num(item.t0)})*lt(t,{_num(item.t1)})"


def _entrance_window(item: Cutaway) -> str:
    end = _num(item.t0 + item.entrance.duration_s)
    return f"gte(t,{_num(item.t0)})*lt(t,{end})"


#: How ``looks`` effects become one ffmpeg fragment for a picture shaped like the
#: base: ``(effects, base info) -> filter fragment``.
EffectCompiler = Callable[[Sequence[Effect], MediaInfo], str]


def compile_effects(
    effects: Sequence[Effect], info: MediaInfo, *, ffmpeg: Optional[str] = None
) -> str:
    """``effects`` as one ffmpeg filter fragment for a picture shaped like ``info``.

    Compiled by ``looks.compile_look`` against the ffmpeg that renders
    (``ffmpeg``, default the one :meth:`CutawayPlan.command` runs): a ``looks``
    step's licence position is a property of the binary, so it is probed
    (``looks.probe``), once per call. :func:`plan_cutaways` calls it only when
    some item has a ``Ripple``; pass ``compile_effects=`` there to compile for
    another binary or to skip the probe.
    """
    import looks
    from looks.spec import ClipSpec

    clip = ClipSpec(
        width=info.width,
        height=info.height,
        fps=info.fps or DEFAULT_FPS,
        duration_s=info.duration or None,
    )
    env = looks.probe(ffmpeg or ffmpeg_exe())
    return looks.vf(
        looks.compile_look(looks.Look(steps=tuple(effects)), clip=clip, env=env)
    )


def _check_source(item: Cutaway, info: MediaInfo) -> None:
    """Refuse a ``src_in`` past the clip's end; warn when the clip runs out early."""
    if item.is_image or not info.duration:
        return
    if item.src_in >= info.duration:
        raise ValueError(
            f"src_in={item.src_in} is at or past the end of {item.src} "
            f"({info.duration:.2f}s)"
        )
    if item.src_in + item.duration > info.duration + SOURCE_OVERRUN_TOLERANCE_S:
        warnings.warn(
            f"{item.src} has only {info.duration - item.src_in:.2f}s after "
            f"src_in={item.src_in}, less than its {item.duration}s window: the "
            "base shows through for the rest of it",
            stacklevel=3,
        )


def plan_cutaways(
    base: str | Path,
    items: Sequence[Cutaway],
    *,
    overlay: str | Path | None = None,
    overlay_xy: tuple[str, str] = DEFAULT_OVERLAY_XY,
    post: Optional[str] = None,
    fit: FitMode = DEFAULT_FIT,
    base_grade: Optional[str] = None,
    insert_grade: Optional[str] = None,
    audio_fade_in_s: float = DEFAULT_AUDIO_FADE_IN_S,
    audio_fade_out_s: float = DEFAULT_AUDIO_FADE_OUT_S,
    ffmpeg_args: Sequence[str] = DEFAULT_FFMPEG_ARGS,
    probe: Callable[[str | Path], MediaInfo] = probe_media,
    compile_effects: EffectCompiler = compile_effects,
) -> CutawayPlan:
    """The :class:`CutawayPlan` of ``items`` over ``base``: nothing is rendered.

    ``base`` runs unbroken (its video timeline and audio); each :class:`Cutaway`
    covers ``[t0, t1)`` with its clip (from ``src_in``) or still image, scaled
    to the base frame by ``fit`` (``"fill"`` crops to cover, ``"fit"``
    letterboxes, ``"stretch"``) and graded by ``insert_grade`` only. An item
    with an ``audio_gain`` has its audio mixed under the base at that linear
    gain. ``overlay`` (an image, e.g. a caption) shows only while an item does,
    at ``overlay_xy`` (ffmpeg expressions); an item with its own ``overlay``
    shows that one instead. An item with a ``transition_in`` enters by it.
    ``post`` (a filter fragment, e.g. ``looks.vf(...)``) runs on the finished
    picture, after every overlay; the warps of items entering by a
    ``looks.Ripple`` run there first (each gated to its own window, compiled
    together by ``compile_effects``). Items are drawn in order, so a later one wins
    where windows overlap.

    Args:
        base: The base video.
        items: The cutaways, at least one.
        overlay: Optional image shown during cutaway windows only (an item's
            own ``overlay`` replaces it during that item).
        overlay_xy: Where ``overlay`` sits (ffmpeg ``overlay`` x, y).
        post: Optional filter fragment on the finished picture.
        fit: ``"fill"`` (default), ``"fit"`` or ``"stretch"``.
        base_grade, insert_grade: ffmpeg filter fragments (e.g. from
            ``looks.vf``) for the base / for the cutaways only.
        audio_fade_in_s, audio_fade_out_s: fades on each mixed cutaway audio.
        ffmpeg_args: Encode arguments (default x264 CRF 18 + AAC).
        probe: How a file's :class:`MediaInfo` is obtained (the only I/O here,
            with the ffmpeg probe of the default ``compile_effects``).
        compile_effects: How the items' ``looks`` effects (a ``Ripple``'s warp)
            become a filter fragment (default :func:`compile_effects`).

    Raises:
        ValueError: no items, or a ``src_in`` past the end of its clip.
    """
    if not items:
        raise ValueError("cutaways needs at least one Cutaway")
    base_info = probe(base)
    warps = [item.warp for item in items if item.warp is not None]
    post = _chain(compile_effects(warps, base_info) if warps else None, post)
    fps = base_info.fps or DEFAULT_FPS
    infos = [probe(item.src) for item in items]

    inputs = ["-i", str(base)]
    fc = [
        "[0:v]"
        + _chain(
            "sidedata=mode=delete:type=DISPLAYMATRIX",  # decode already rotated it
            f"fps={_num(fps)}",
            f"scale={base_info.width}:{base_info.height}",
            "setsar=1",
            base_grade,
            "format=yuv420p",
        )
        + "[b0]"
    ]
    last, mixed = "b0", []
    for i, (item, info) in enumerate(zip(items, infos), start=1):
        _check_source(item, info)
        if base_info.duration and item.t0 >= base_info.duration:
            warnings.warn(
                f"cutaway window starts at {item.t0}s, after the base ends "
                f"({base_info.duration:.2f}s): it will not show",
                stacklevel=3,
            )
        entrance_end = item.t0 + (item.entrance.duration_s if item.entrance else 0.0)
        if base_info.duration and item.t0 < base_info.duration < entrance_end:
            warnings.warn(
                f"cutaway entrance runs to {entrance_end:g}s, past the base's end "
                f"({base_info.duration:.2f}s): it is cut short there",
                stacklevel=3,
            )
        inputs += _input_args(item, fps)
        fc.append(
            _item_video(
                i,
                item,
                base=base_info,
                source=info,
                fps=fps,
                fit=fit,
                grade=insert_grade,
            )  # fmt: skip
        )
        shown = f"v{i}"
        if item.entrance is not None:
            entrance, shown = _entrance(i, item, fps=fps)
            fc += entrance
            fc.append(
                f"[{last}][x{i}]overlay=eof_action=pass:"
                f"enable='{_entrance_window(item)}'[e{i}]"
            )
            last = f"e{i}"
        window = _window(item, after_entrance=True)
        fc.append(f"[{last}][{shown}]overlay=eof_action=pass:enable='{window}'[b{i}]")
        last = f"b{i}"
        if item.audio_gain is not None and not item.is_image and info.has_audio:
            fc.append(
                _item_audio(
                    i, item, fade_in_s=audio_fade_in_s, fade_out_s=audio_fade_out_s
                )
            )
            mixed.append(f"[a{i}]")

    # one overlay per (image, position): the plan's on the items without their
    # own, each item's own on its window
    captions: dict[tuple[str, tuple[str, str], bool], list[Cutaway]] = {}
    for item in items:
        mine = item.overlay is not None
        image = item.overlay if mine else overlay
        if image is not None:
            xy = tuple(item.overlay_xy or overlay_xy)
            captions.setdefault((str(image), xy, mine), []).append(item)
    own = [item for item in items if item.overlay is not None]
    for k, ((image, (x, y), mine), shown_on) in enumerate(captions.items()):
        inputs += ["-loop", "1", "-i", image]
        windows = "+".join(_window(item) for item in shown_on)
        if not mine and own:
            # the plan's caption steps aside wherever an item shows its own
            windows = f"({windows})*not({'+'.join(_window(i) for i in own)})"
        # the finished picture is ``[vout]`` (after the last overlay, or ``post``)
        label = "vout" if k == len(captions) - 1 and not post else f"c{k}"
        fc.append(f"[{len(items) + 1 + k}:v]format=rgba[cap{k}]")
        fc.append(
            f"[{last}][cap{k}]overlay='{x}':'{y}':shortest=1:enable='{windows}'"
            f"[{label}]"
        )
        last = label
    if post:
        fc.append(f"[{last}]{post}[vout]")
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
        encode_args=tuple(ffmpeg_args),
    )


# --- execution ------------------------------------------------------------------


def render_cutaways(
    plan: CutawayPlan, *, default_path: str | Path, output: Output = None
):
    """Run ``plan`` with ffmpeg; ``output`` is the usual file-first egress target."""

    def write(path: Path) -> None:
        proc = subprocess.run(plan.command(path), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-2000:]}")

    return write_egress(output, default_path=default_path, write=write)


def cutaways(
    base: str | Path,
    items: Sequence[Cutaway],
    *,
    output: Output = None,
    **plan_options,
) -> Path:
    """Cut away from ``base`` to ``items`` at their windows, in one ffmpeg pass.

    :func:`plan_cutaways` then :func:`render_cutaways`; ``plan_options`` are
    :func:`plan_cutaways`'s (``overlay``, ``fit``, ``insert_grade``, ...), which
    documents them. To inspect before running, call ``plan_cutaways`` yourself
    (``plan.filter_complex``, ``plan.command(out)``).

    Args:
        base: The base video.
        items: The :class:`Cutaway` windows.
        output: Where to put the result: ``None`` (``<base>_cutaways.<ext>``
            beside the base), a file path, a directory, or a callable sink.

    Returns:
        The output path (or the sink's return value).
    """
    plan = plan_cutaways(base, items, **plan_options)
    base = Path(base)
    return render_cutaways(
        plan,
        default_path=base.with_name(f"{base.stem}_cutaways{base.suffix}"),
        output=output,
    )
