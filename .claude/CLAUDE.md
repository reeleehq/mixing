# mixing — agent & contributor guide

`mixing` is a toolkit for **video and audio editing** (slicing, fades, audio
replace/mix, Ken Burns, thumbnails, subtitles, speech-to-text + filler removal,
TTS dubbing, chapter detection). It is **AI-first**: the primary users are AI
agents, so the fastest way to use it well is through the bundled skills and
agents.

## Start here (AI-first)

- **Skills** (`.claude/skills/`) — task-oriented playbooks for *using* mixing:
  - [`mixing`](skills/mixing/SKILL.md) — router/overview: architecture, the
    `output` egress protocol, optional extras, ffmpeg. Read this first.
  - [`mixing-audio`](skills/mixing-audio/SKILL.md) — `Audio`, fades, crop,
    concat, overlay, normalize, alignment, segmentation.
  - [`mixing-video`](skills/mixing-video/SKILL.md) — `Video`, crop/loop/speed,
    replace/normalize audio, Ken Burns, thumbnails, subtitles, dimension presets.
  - [`mixing-transcript`](skills/mixing-transcript/SKILL.md) — transcribe
    (ElevenLabs Scribe), filler removal, SRT/prose, chapters.
  - [`mixing-dubbing`](skills/mixing-dubbing/SKILL.md) — TTS re-voicing and
    SRT translation.
- **Agents** (`.claude/agents/`):
  - [`mixing-editor`](agents/mixing-editor.md) — performs an editing task with
    mixing end-to-end (use for "edit this media …").
  - [`mixing-dev`](agents/mixing-dev.md) — works *on* the mixing codebase
    (use for changes/refactors here).

The rest of this file is the architectural contract those skills/agents rely on.

## Architecture

Focused subpackages, each with a clear dependency footprint:

| Module | Purpose | Heavy deps |
|---|---|---|
| `mixing.audio` | `Audio`/`AudioSamples`, fades, crop, concat, overlay, `loop_audio`, `duck_audio`, `find_audio_offset`, segmentation | `pydub`, `numpy`, `scipy` |
| `mixing.audio.beats` | `beat_grid`/`BeatGrid` — beat/downbeat/onset analysis | `librosa` (extra `beats`, lazy-imported) |
| `mixing.audio.tagging` | `WindowProbabilities` + `ast_audioset_classifier`: the classifier seam behind `find_segments(strategy="audioset")` (songs vs jingles vs talk on radio shows) | `torch`, `transformers` (extra `ml`, lazy-imported) |
| `mixing.video` | `Video`/`VideoFrames`, crop/loop/speed, `replace_audio`, `normalize_audio`, `overlay_ambient_bed`, `ken_burns_*`, `concatenate_videos`, thumbnails, subtitles | `moviepy`, `opencv`, `pillow`, `looks` (vocabulary only — stdlib) |
| `mixing.video.genai` | Google Vertex AI **Veo** generation | `google-genai` (extra `gen`) |
| `mixing.transcript` | ElevenLabs **Scribe** STT (stdlib HTTP, cached), filler removal, SRT/prose | stdlib only |
| `mixing.dubbing` | ElevenLabs **TTS** re-voice / translate (stdlib HTTP, cached) | stdlib (`dub_video_from_srt` needs `moviepy`) |
| `mixing.srt` | **canonical** SRT/timeline `Cue` + parse/format/shift | pure stdlib |
| `mixing.chapters` | transcript → `Chapter` markers (LLM-pluggable) | pure (LLM via `aix`, extra `llm`) |
| `mixing.egress` | the `output` protocol (see below) | pure |
| `mixing.util` | `has_ffmpeg`, `to_seconds`, `require_package`, clipboard | pure |
| `mixing._cache`, `mixing._elevenlabs` | internal: shared disk cache + ElevenLabs auth | pure |

### `looks` owns the geometry vocabulary; `mixing` owns the implementations

`mixing.video.video_util` **imports** `SOCIAL_SIZES`, the
`stretch`/`fit`/`fill` mode names (`looks.geometry.FitMode`) and the `social`
backdrop's `DFLT_BACKDROP_BLUR_SIGMA` / `DFLT_BACKDROP_DIM` from
`looks.geometry` rather than declaring them. That is the whole of the edge — a
**vocabulary port, not a code move**:

- The arithmetic stays here. `video_util` imports moviepy at module scope, and
  the `social` branch is not arithmetic at all: it composites a scaled,
  centre-cropped, Gaussian-blurred, dimmed copy of the input behind the fitted
  foreground. `looks.geometry` is pure, total, media-library-free arithmetic.
- **`get_video_dimensions` does not move.** It is a *probe* (it opens a file, or
  reads a live clip), not geometry, and `paces` calls it at five sites through
  the top-level facade behind an explicit `mixing` floor.
- `looks` declares **no dependencies**, so the edge is free. Don't let it grow
  into a runtime dependency on `looks`' ffmpeg/compile layers without saying so.

`mixing/tests/test_guard_video_util.py` pins this: identity with `looks`' dict
(equality would let two copies drift), and — because the imported constants
*equal* the literals they replaced — two render-level tests that monkeypatch the
constants and assert the pixels move.

### Lazy facade — keep light imports light

`mixing/__init__.py` is a **PEP 562 lazy facade**: `import mixing` and
`import mixing.chapters` / `import mixing.srt` pull **none** of
moviepy/opencv/pydub/PIL. Heavy backends load only when a name that needs them
is first accessed (`mixing.Video`, `from mixing.video import replace_audio`).
`mixing/dubbing/__init__.py` is lazy the same way (TTS/SRT stay light;
`dub_video_from_srt` pulls moviepy on access).

**Rule when adding code:** never add an eager top-of-module
`import moviepy`/`cv2` to a module that light code imports. Put heavy imports
inside the function/method, or behind the lazy facade. There are tests asserting
`import mixing` stays light — keep them green.

## The `output` egress protocol — the central convention

Every result-producing function takes a single **`output`** parameter whose
*role* is constant ("what to do with the result") while its *type* is open
(implemented in `mixing.egress`):

| `output` | behavior |
|---|---|
| `None` | object producers → return the in-memory object; file producers → save beside the input and return the `Path` |
| a **file path** | write there; return the `Path` |
| a **directory** | write with an auto-derived name; return the `Path` |
| a **callable** | return `output(result)` — the general sink/escape hatch |

Use the helpers, don't reinvent them:
- `mixing.egress.deliver(result, output, *, write, default_name)` — *object-first*
  (None returns the object). For audio editing functions that build a value.
- `mixing.egress.write_egress(output, *, default_path, write)` — *file-first*
  (None saves to `default_path`). For file→file operations.

**Naming rules:**
- The single/primary destination is always **`output`**.
- Only when a function emits **multiple** artifacts do you qualify them
  (`remove_fillers(..., output_dir=, output_media=)`). One destination → `output`.
- Inputs use domain nouns (`video`, `audio`, `media`, `transcript`).
- Arguments beyond the 2nd–3rd position are **keyword-only**.
- No magic numbers — name them as module constants (e.g.
  `SECONDS_PER_CHAPTER_HEURISTIC`, `DEFAULT_FRAME_TIME_FRACTION`).

## Conventions

- Favor functional style; small focused helpers (`_underscore` for
  module-private, inner functions for single-use). `dataclasses` for data.
- Every module needs a top-level docstring (auto-extracted for docs).
- Informative errors; for optional deps, `mixing.util.require_package` or a
  clear `ImportError` pointing at the right extra.
- Time is always **seconds** (`float`) except SRT timestamp strings. `Audio`
  also supports `time_unit` of `seconds|milliseconds|samples`; `Video` supports
  `seconds|frames`.

## Optional extras

The base install carries only the pure modules (`srt`, `chapters`, `errors`,
`util`, `egress`, `transcript`, TTS). **`media`** supplies the backends
`mixing.audio` / `mixing.video` import (moviepy, opencv-contrib-python, scipy,
pydub, numpy, pillow, burns, looks); importing either without it raises an
`ImportError` naming `pip install 'mixing[media]'` (`util.require_media_extra`,
called at the top of both package `__init__`s). Tests install it via `testing`.

`pip install mixing[<extra>]`: `media` (see above), `audio` (ffmpeg-python/soundfile, on top of `media`),
`widget` (Jupyter audio widget), `gen` (Veo/`google-genai`), `llm` (`aix`, for
chapter titling + SRT translation), `clipboard` (`pyclip`), `beats` (`librosa`,
for `mixing.audio.beats.beat_grid`), `ml` (`torch` + `transformers`, for the `"audioset"` segmentation strategy), `dev`/`testing`.
Tests needing a model download are marked `slow` and skip unless `MIXING_RUN_SLOW=1`.
**ffmpeg must be on PATH** for most real work (`mixing.has_ffmpeg()` checks).

## Tests = guardrails

Run `python -m pytest -q` (that's also how you get the current test count — don't
pin a number here, it only drifts). Many are *characterization* tests in
`mixing/tests/test_guard_*.py` that pin exact current behavior. **Do not change a
guardrail assertion to make a refactor pass** — if behavior must change,
that's a deliberate decision: change the assertion *and* say why in the commit.
Shared synthetic-media fixtures live in `conftest.py`
(`make_tone_audio`/`tone_audio`, `make_color_video`/`color_video`).

Demos/examples live in `misc/` (never inside the importable package).
