# mixing

Tools for **video and audio editing** in Python — slicing, fades, audio
replace/mix, Ken Burns, thumbnails, subtitles, speech-to-text + filler removal,
TTS dubbing, and chapter detection.

```bash
pip install 'mixing[media]'   # audio + video editing (moviepy, opencv, scipy, pydub)
pip install mixing            # base: only the pure modules (srt, chapters, errors, util, egress, transcript, TTS)
```

The base install is light on purpose: it carries no media backend (and so no bundled ffmpeg binary). `mixing.audio` and `mixing.video` need the `media` extra; importing them without it raises an `ImportError` that says so.

---

## 🤖 AI-first: start with the skills & agents

`mixing` is built to be driven by AI coding agents. The fastest, most reliable
way to use it is through the bundled **skills** and **agents** in
[`.claude/`](.claude/) — task-oriented playbooks that encode the right
functions, the calling conventions, and the gotchas, so an agent gets it right
the first time.

**Skills** (`.claude/skills/`) — *how to use mixing for a task*:

| Skill | Use it for |
|---|---|
| [`mixing`](.claude/skills/mixing/SKILL.md) | **Start here** — router, architecture, the `output` protocol, extras |
| [`mixing-audio`](.claude/skills/mixing-audio/SKILL.md) | slice, fade, crop, concat, overlay, normalize, align, segment audio |
| [`mixing-video`](.claude/skills/mixing-video/SKILL.md) | slice, crop, loop, speed, replace/normalize audio, Ken Burns, thumbnails, subtitles |
| [`mixing-transcript`](.claude/skills/mixing-transcript/SKILL.md) | transcribe, remove fillers, SRT/prose, chapters |
| [`mixing-dubbing`](.claude/skills/mixing-dubbing/SKILL.md) | TTS re-voicing & SRT translation |

**Agents** (`.claude/agents/`):

- [`mixing-editor`](.claude/agents/mixing-editor.md) — performs an editing task
  end-to-end ("edit this media …").
- [`mixing-dev`](.claude/agents/mixing-dev.md) — works *on* the mixing codebase.

Using Claude Code? These are discovered automatically from `.claude/`. The
architectural contract they share lives in [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

---

## Quick start

```python
import mixing  # cheap: no moviepy/opencv loaded yet

assert mixing.has_ffmpeg()  # most ops need ffmpeg on PATH

# --- audio ---
from mixing.audio import Audio

Audio("song.mp3")[10:30].fade_in(2).fade_out(3).save(output="clip.mp3")

# --- video ---
from mixing.video import replace_audio, ken_burns_video

replace_audio("clip.mp4", "music.mp3", mix_ratio=0.7, output="mixed.mp4")
ken_burns_video("cover.jpg", duration=8, output="cover.mp4")

# --- transcript: remove "uh"/"um" (ElevenLabs Scribe + ffmpeg) ---
from mixing.transcript import remove_fillers

result = remove_fillers("talk.mov", "out/")  # uses $ELEVENLABS_API_KEY
print(result.cleaned_media)
```

## Core ideas (learn once)

**The `output` protocol.** Every result-producing function takes one `output`
argument whose *role* is constant and *type* is open:

```python
crop_video("in.mp4", 5, 15, output="out.mp4")  # file → writes it, returns Path
crop_video("in.mp4", 5, 15, output="clips/")  # dir  → auto-named file inside
crop_video("in.mp4", 5, 15)  # None → saves beside the input
fade_in("in.mp3", output=lambda audio: audio)  # callable → receives the result
```

Object producers (audio editing) return the in-memory object when `output=None`,
so you can chain and save when ready. Functions that emit *several* artifacts
qualify the destinations (`remove_fillers(media, output_dir=…, output_media=…)`).

**Lazy & tiered.** `import mixing` (and `import mixing.chapters` /
`mixing.srt`) pull **no** heavy backends; moviepy/opencv/pydub load only when you
touch a name that needs them. Import the facade (`mixing.Video`) or the
subpackage (`from mixing.video import Video`).

**Sliceable media.** `Audio` and `Video` are lazy, sliceable, context-managed
views:

```python
from mixing.video import Video

with Video("movie.mp4") as v:
    v[10:30].save(output="cut.mp4")  # seconds; v[100] returns a single frame
```

**Logging, not printing.** `mixing` never writes progress messages to stdout —
every module logs through `logging.getLogger(__name__)`, and the package root
carries a `NullHandler`, so `mixing` is silent by default. This keeps library
calls usable from a stdout-parsing consumer. To see mixing's own progress
messages (e.g. while working interactively, or when an operation like
`mixing.video.genai.generate_video` fails and you want the full diagnostic),
opt in with:

```python
import logging

logging.basicConfig(level=logging.INFO)  # or level=logging.ERROR for errors only
```

## Requirements

- **ffmpeg** on PATH (most operations). macOS: `brew install ffmpeg`;
  Debian/Ubuntu: `sudo apt install ffmpeg`; Windows: download from
  [ffmpeg.org](https://ffmpeg.org/download.html) and add `bin` to PATH.
- **ElevenLabs API key** for `mixing.transcript` / `mixing.dubbing`
  (`ELEVENLABS_API_KEY` or `ELEVEN_API_KEY`, or pass `api_key=`). Responses are cached on disk, so
  re-runs are free and offline.
- **Google Cloud** auth for AI video generation
  (`mixing.video.genai`, Vertex AI Veo).

### Optional extras

```bash
pip install mixing[media]      # moviepy / opencv / scipy / pydub: mixing.audio + mixing.video
pip install mixing[audio]      # an alias: media + soundfile (kept for existing installs)
pip install mixing[widget]     # interactive Jupyter audio widget
pip install mixing[gen]        # Google Vertex AI Veo generation
pip install mixing[llm]        # aix — chapter titling + SRT translation
pip install mixing[clipboard]  # get file paths from the clipboard
pip install mixing[beats]      # librosa — beat/downbeat/onset analysis
pip install mixing[ml]         # torch + transformers — find_segments(strategy="audioset"): songs vs jingles vs talk
```

## What's inside

| Subpackage | Highlights |
|---|---|
| `mixing.audio` | `Audio`, `fade_in/out`, `crop_audio`, `concatenate_audio`, `overlay_audio`, `loop_audio`, `duck_audio`, `find_audio_offset`, `find_segments`/`extract_segments` (incl. `strategy="audioset"`, extra `ml`), `text_before_segments`, `find_talk_edges` (host talk at a song's start/end), `beat_grid`/`BeatGrid` (extra `beats`), `loudness` (BS.1770 LUFS), `level_turns` (per-speaker loudness matching), `process_turns`, `voice_profile` (per-speaker f0/loudness/timbre), `shift_pitch` (formant-preserving with rubberband) |
| `mixing.video` | `Video`, `crop_video`, `loop_video`, `change_speed`, `replace_audio`, `normalize_audio`, `overlay_ambient_bed`, `ken_burns_video`/`ken_burns_film`, `concatenate_videos` (+ transitions: `crossfade_transition`, `fade_through_black`, …), `make_thumbnail`, `write_subtitles_in_video`, `SOCIAL_SIZES` |
| `mixing.video.genai` | `generate_video` (Vertex AI Veo) |
| `mixing.transcript` | `transcribe`, `remove_fillers`, `srt_for_media`, `words_to_srt`/`words_to_prose` |
| `mixing.chapters` | `detect_chapters`, `Chapter` |
| `mixing.dubbing` | `text_to_speech`, `list_voices`, `translate_srt`, `dub_video_from_srt` |
| `mixing.srt` | `Cue`, `parse_srt`, `dump_srt`, `seconds_to_srt_time` |

See the [skills](.claude/skills/) for the full, example-driven API of each.

## License

MIT
