# mixing.ffmpeg

Run one ffmpeg process with a watchdog: the runner `mixing.decode` and
[`mixing.probe`](mixing.probe.html.md#module-mixing.probe) share.

[`run_ffmpeg()`](#mixing.ffmpeg.run_ffmpeg) is a context manager around `subprocess.Popen`: ffmpeg’s
stderr goes to a temporary file (a pipe nobody reads fills up and blocks it),
`timeout=` arms a watchdog that kills ffmpeg **and what it started** (a
wrapper script or a Windows shim would otherwise leave the real ffmpeg running,
holding the pipes open) and raises [`FfmpegTimeout`](#mixing.ffmpeg.FfmpegTimeout), and any exception in
the block kills it. Pure stdlib, so importing it pulls no media library.

```pycon
>>> from mixing.ffmpeg import run_ffmpeg
>>> with run_ffmpeg([exe, "-i", "talk.mp4"], what="read", timeout=5, check=False) as p:
...     pass
>>> "Duration:" in p.message
True
```

### Functions

| [`run_ffmpeg`](#mixing.ffmpeg.run_ffmpeg)(cmd, \*, what[, stdin, stdout, ...])   | Run the ffmpeg `cmd` for the block; yield the process.                                                                                     |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [`check_timeout`](#mixing.ffmpeg.check_timeout)(timeout)                            | `timeout` checked: `None` (or `inf`) for no bound, else seconds > 0.                                                                       |
| [`ignore_closed_stdin`](#mixing.ffmpeg.ignore_closed_stdin)()                             | Ignore writing to (or closing) the stdin of an ffmpeg that stopped reading: a broken pipe, or `EINVAL` on Windows; its exit code says why. |

### Exceptions

| [`FfmpegTimeout`](#mixing.ffmpeg.FfmpegTimeout)   | ffmpeg ran past its `timeout` and was killed.   |
|------------------------------------------------------------------|-------------------------------------------------|

### *exception* mixing.ffmpeg.FfmpegTimeout

Bases: [`TimeoutError`](https://docs.python.org/3/builtins/exceptions.html#TimeoutError), [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

ffmpeg ran past its `timeout` and was killed.

A `TimeoutError`, and a `RuntimeError` like every other ffmpeg failure
here, so a caller catching either sees it.

### mixing.ffmpeg.check_timeout(timeout)

`timeout` checked: `None` (or `inf`) for no bound, else seconds > 0.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### mixing.ffmpeg.ignore_closed_stdin()

Ignore writing to (or closing) the stdin of an ffmpeg that stopped
reading: a broken pipe, or `EINVAL` on Windows; its exit code says why.

### mixing.ffmpeg.run_ffmpeg(cmd, , what, stdin=-3, stdout=-3, timeout=None, check=True)

Run the ffmpeg `cmd` for the block; yield the process.

Its stderr goes to a temporary file (a pipe nobody reads fills up and
blocks ffmpeg); once it is done the text is `proc.message`. Any
exception in the block (a generator closed early included) kills it. On
the way out its pipes are closed, a broken stdin pipe ignored (ffmpeg
stopped reading: its exit code says why), and it is waited for. With
`check` a non-zero exit raises `RuntimeError("ffmpeg could not {what}:
...")` with the tail of its message. With `timeout` (seconds, see
[`check_timeout()`](#mixing.ffmpeg.check_timeout)) a watchdog kills it, and what it started, that long
after it started, waiting included, and [`FfmpegTimeout`](#mixing.ffmpeg.FfmpegTimeout) is raised
(the block sees an ordinary end of stream or broken pipe); an ffmpeg that
finished in time is never reported as timed out.
