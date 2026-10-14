# mixing.errors

Typed errors `mixing` raises — the home for exceptions a caller may want to catch.

Pure stdlib, imported by the lazy facade without pulling any backend, so a consumer
can `from mixing import WindowTooWideForClip` in a light module and still catch what
a heavy one raises.

The convention: an error class carries the MEASURED numbers as attributes, not only in
its message, so a caller can act on them (retry at the largest valid window, say) rather
than parse prose.

### Exceptions

| [`MixingError`](#mixing.errors.MixingError)                               | Base for every error `mixing` raises on its own behalf.                |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`WindowTooWideForClip`](#mixing.errors.WindowTooWideForClip)(\*, clip_index, ...) | An explicit analysis window leaves a clip no second, independent look. |

### *exception* mixing.errors.MixingError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base for every error `mixing` raises on its own behalf.

### *exception* mixing.errors.WindowTooWideForClip(, clip_index, clip_duration_s, window_s, hop_s, max_window_s)

Bases: [`MixingError`](#mixing.errors.MixingError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

An explicit analysis window leaves a clip no second, independent look.

Raised by `mixing.audio.align_clips_to_reference()` when the caller passes a
`window_s` wider than [`max_window_s`](#mixing.errors.WindowTooWideForClip.max_window_s) — the length above which the clip holds
no second INDEPENDENT window, so the consensus vote has nothing to arbitrate and
`support` comes back `None`.

Why an error and not a clamp or a silent single window: the caller asked for a
windowed vote and would otherwise receive a single whole-clip correlation whose
only signal is the ABSENCE of a support number — indistinguishable from a clip
that is genuinely too short to support at any window (issue #43). Clamping would
be worse still: an explicit `window_s` is the caller saying what a window means
for their material, and silently measuring at a different one makes `support`
incomparable across a set for a reason nothing reports.

It subclasses [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) because it is an argument that cannot be honored,
so existing `except ValueError` handlers around alignment keep working.

#### clip_index

Position of the offending clip in the `clips` sequence, or
`None` when the caller did not identify one.

#### clip_duration_s

The clip’s duration in seconds.

#### window_s

The window that was asked for.

#### hop_s

The hop in force — the caller’s, or the one derived from `window_s`.
Reported because it is part of the grid that was asked for; it is NOT part
of the bound, which does not depend on it (see
`_max_supportable_window_s()`).

#### max_window_s

The largest window that still leaves this clip a second,
independent look. The bound is inclusive, so for any clip long enough to be
worth aligning a retry at exactly this window measures — which is what makes
it worth reporting. A degenerate clip of a sample or two has no such window
at all; the value is floored at one sample there and is a lower bound rather
than a promise.
