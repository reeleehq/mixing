"""Typed errors `mixing` raises — the home for exceptions a caller may want to catch.

Pure stdlib, imported by the lazy facade without pulling any backend, so a consumer
can ``from mixing import WindowTooWideForClip`` in a light module and still catch what
a heavy one raises.

The convention: an error class carries the MEASURED numbers as attributes, not only in
its message, so a caller can act on them (retry at the largest valid window, say) rather
than parse prose.
"""

import math

__all__ = ["MixingError", "WindowTooWideForClip", "ProbeError", "MediaNotFoundError"]

#: Decimal places the remedy in a :class:`WindowTooWideForClip` message is stated to.
#: The number is rounded DOWN to them, so what the message tells a caller to pass is
#: always inside the bound rather than one rounding step outside it.
_ADVICE_DECIMALS = 6


class MixingError(Exception):
    """Base for every error `mixing` raises on its own behalf."""


class ProbeError(MixingError, ValueError):
    """ffmpeg/ffprobe could not read a media file (see :func:`mixing.probe.probe`).

    A :class:`ValueError`, so the ``except ValueError`` handlers that caught the
    probes it replaced keep working.
    """


class MediaNotFoundError(ProbeError, FileNotFoundError):
    """The media file to probe does not exist (also a :class:`FileNotFoundError`)."""


class WindowTooWideForClip(MixingError, ValueError):
    """An explicit analysis window leaves a clip no second, independent look.

    Raised by :func:`mixing.audio.align_clips_to_reference` when the caller passes a
    ``window_s`` wider than :attr:`max_window_s` — the length above which the clip holds
    no second INDEPENDENT window, so the consensus vote has nothing to arbitrate and
    ``support`` comes back ``None``.

    Why an error and not a clamp or a silent single window: the caller asked for a
    windowed vote and would otherwise receive a single whole-clip correlation whose
    only signal is the ABSENCE of a support number — indistinguishable from a clip
    that is genuinely too short to support at any window (issue #43). Clamping would
    be worse still: an explicit ``window_s`` is the caller saying what a window means
    for their material, and silently measuring at a different one makes ``support``
    incomparable across a set for a reason nothing reports.

    It subclasses :class:`ValueError` because it is an argument that cannot be honored,
    so existing ``except ValueError`` handlers around alignment keep working.

    Attributes:
        clip_index: Position of the offending clip in the ``clips`` sequence, or
            ``None`` when the caller did not identify one.
        clip_duration_s: The clip's duration in seconds.
        window_s: The window that was asked for.
        hop_s: The hop in force — the caller's, or the one derived from ``window_s``.
            Reported because it is part of the grid that was asked for; it is NOT part
            of the bound, which does not depend on it (see
            :func:`~mixing.audio.audio_ops._max_supportable_window_s`).
        max_window_s: The largest window that still leaves this clip a second,
            independent look. The bound is inclusive, so for any clip long enough to be
            worth aligning a retry at exactly this window measures — which is what makes
            it worth reporting. A degenerate clip of a sample or two has no such window
            at all; the value is floored at one sample there and is a lower bound rather
            than a promise.
    """

    def __init__(
        self,
        *,
        clip_index: "int | None",
        clip_duration_s: float,
        window_s: float,
        hop_s: float,
        max_window_s: float,
    ) -> None:
        self.clip_index = clip_index
        self.clip_duration_s = float(clip_duration_s)
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.max_window_s = float(max_window_s)
        who = "clip" if clip_index is None else f"clip {clip_index}"
        # Rounded DOWN, never to nearest: a ceiling printed even one ULP above the real
        # one sends the caller straight back into this exception, which is the one
        # remedy the message must not suggest.
        advice = math.floor(self.max_window_s * 10**_ADVICE_DECIMALS) / (
            10**_ADVICE_DECIMALS
        )
        super().__init__(
            f"window_s={self.window_s:.3f} is wider than {who} "
            f"({self.clip_duration_s:.3f} s) allows: no second INDEPENDENT window fits, "
            f"so the consensus vote has nothing to arbitrate and support would be None "
            f"for a reason nothing reports — pass window_s <= "
            f"{advice:.{_ADVICE_DECIMALS}f}, or window_s=None to fit the window to each "
            f"clip. "
            f"(hop_s={self.hop_s:.3f} does not enter this bound: the analysis grid "
            f"always ends with a window one window's length from the clip's end, "
            f"whatever the hop.)"
        )
