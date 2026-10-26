---
name: mixing-transcript
description: >
  Transcribe speech to text, remove filler words (ums/uhs), build SRT subtitles
  or clean prose, and detect chapter markers — using the `mixing` package's
  `transcript` and `chapters` modules (ElevenLabs Scribe under the hood). Trigger
  whenever someone wants to "transcribe this audio/video", "get an SRT/captions",
  "remove the ums and uhs", "clean up the fillers", "make chapters/timestamps",
  "detect topic shifts", or "get word-level timestamps" from a media file. For
  re-voicing/translation use mixing-dubbing; for plain audio/video edits use
  mixing-audio / mixing-video. Read the `mixing` router skill first for the
  `output` egress protocol and ffmpeg/key setup.
---

# mixing-transcript — STT, filler removal, SRT/prose, chapters

Speech-to-text via **ElevenLabs Scribe**, filler-word cuts, SRT/prose rendering,
and transcript→chapter detection. `import mixing` is light; pull what you need.

## Quick start

```python
import mixing

assert mixing.has_ffmpeg(), "install ffmpeg (brew install ffmpeg / apt install ffmpeg)"

from mixing.transcript import srt_for_media, remove_fillers, transcribe

# 1. Captions for a media file (transcribe once, persist beside it, reuse).
srt_text, srt_path = srt_for_media("talk.mp4")  # writes talk.srt

# 2. Strip the ums/uhs and write cleaned media + transcripts into a folder.
result = remove_fillers("talk.mp4", "out/")  # -> FillerRemovalResult
print(result.cleaned_media)  # out/talk.cleaned.mp4

# 3. Word-level timestamps as raw data.
resp = transcribe("talk.mp4", cache=True)  # dict with resp["words"]
```

**Needs `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`)** (env var, or pass `api_key=`). Responses are cached
on disk, so re-running the same input is free and offline (`cache=`/`refresh=`).

## Key functions

### transcribe — raw Scribe response
```python
from mixing.transcript import transcribe

resp = transcribe(
    "talk.mp4",  # path (audio OR video) or raw bytes
    cache=True,  # on-disk cache; True=default dir, or pass a path
    # api_key=..., model_id="scribe_v1",
    # timestamps_granularity="word",  # "word"(default)|"character"|"none"
    # diarize=False, language_code="en", refresh=False,
)
resp["words"]  # [{"text","start","end","type","confidence"}, ...]
resp["text"]  # full plain text
resp["audio_duration_secs"]
```
`type` is `"word"`, `"spacing"`, or `"audio_event"` (e.g. `(laughs)` when
`tag_audio_events=True`, the default). Times are **seconds** (float).
`transcribe` returns a dict — it is NOT an `output`-protocol producer.
`default_cache_dir()` returns the cache location (honors
`$MIXING_TRANSCRIPT_CACHE_DIR`, `$XDG_CACHE_HOME`, then `~/.cache/mixing/transcript`).

### srt_for_media — transcribe-once-persist-reuse
```python
from mixing.transcript import srt_for_media

srt_text, srt_path = srt_for_media(
    "talk.mp4",
    # srt_path=None,    # default: media path with .srt suffix
    # reuse=True,       # if the .srt already exists, read it (no API call)
    # refresh=False,    # force re-transcription, overwrite the .srt
    # cache=True,       # pass-through to transcribe's response cache
    # max_chars=80,     # max chars per cue
    # language_code="en", diarize=False,  # forwarded to transcribe
)
```
Returns `(srt_text, Path)`. Existing/hand-corrected SRTs are reused untouched.

### remove_fillers — end-to-end ums/uhs removal
```python
from mixing.transcript import remove_fillers

result = remove_fillers(
    "talk.mp4",  # input media (video or audio)
    "out/",  # output_dir (created if missing) for all artifacts
    # output_media=None,   # cleaned media; default out/<stem>.cleaned.mp4
    # fillers=DEFAULT_FILLER_TOKENS,        # override the filler set
    # audio_events=DEFAULT_AUDIO_EVENTS_TO_CUT,  # default removes (coughs); keeps (laughs)
    # api_key=..., scribe_kwargs={...}, apply_keeps_kwargs={...},
    # scribe_data=None,    # replay a saved Scribe dict — NO network call
)
result.cleaned_media  # Path to cleaned media
result.cuts  # [{"start","end","label"}, ...]   removed ranges
result.keeps  # [{"start","end"}, ...]           kept ranges
# also: transcript_md/srt, cleaned_md/srt, scribe_json, cuts_json, keeps_json, duration
```
This emits **multiple artifacts**, so it qualifies its destinations
(`output_dir=` + `output_media=`) rather than a single `output`.
**Offline replay:** pass `scribe_data=json.load(open("out/scribe.json"))` to
re-cut with no API call — perfect for tweaking `fillers=` cheaply.

### detect_chapters — transcript → chapter markers
```python
import mixing  # eager facade export

chapters = mixing.detect_chapters(
    resp,  # a Scribe dict, OR resp["words"], OR SRT text, OR a cue list
    # duration=None,       # inferred from last timestamp if omitted
    # min_chapters=3, max_chapters=8, min_spacing=10.0,  # players need >=10s spacing
    # target_count=None,   # ~1 chapter / 90s, clamped to [min,max]
    # segment_fn=None,     # default uses an LLM (aix); see below to avoid it
    # model=None,
)
for ch in chapters:  # list[mixing.Chapter]
    print(ch.start, ch.title)  # (start_seconds: float, title: str)
```
Returns `[]` when media is too short to host `min_chapters` spaced markers (first
chapter is always forced to `0:00`). Output is target-neutral — formatting into
YouTube/PSC/ID3 is a publication-layer job (e.g. the `yb` package).
`detect_chapters`/`Chapter` are also at `from mixing.chapters import ...`.

**Avoid the LLM dep** ([llm]/`aix`) by passing your own `segment_fn(segments, target_count) -> [{"start","title"}]`:
```python
def first_sentences(segments, target_count):  # naive, no LLM
    step = max(1, len(segments) // target_count)
    return [{"start": s["start"], "title": s["text"][:50]} for s in segments[::step]]


mixing.detect_chapters(resp, segment_fn=first_sentences)
```

## Building blocks (compose your own pipeline)
```python
from mixing.transcript import (
    is_filler,
    build_cuts,
    keeps_from_cuts,  # cut math
    words_to_srt,
    words_to_prose,
    words_to_srt_remapped,
    remap_time_after_cuts,
    extract_audio,
    apply_keeps,  # ffmpeg helpers
)

words = transcribe("talk.mp4", cache=True)["words"]
cuts = build_cuts(words)  # [{"start","end","label"}]
keeps = keeps_from_cuts(cuts, duration=resp["audio_duration_secs"])

words_to_srt(words, max_chars=80)  # SRT (no removal, no remap)
words_to_prose(words, drop_fillers=True)  # clean paragraphs
words_to_srt_remapped(words, cuts)  # SRT aligned to the post-cut timeline
apply_keeps("talk.mp4", "cut.mp4", keeps)  # ffmpeg re-encode -> Path
extract_audio("talk.mp4", "talk.mp3")  # mono 16k mp3 for STT -> Path
```

## Recipes

- **Just the captions:** `srt_for_media("v.mp4")` — reuses an existing `.srt`.
- **Clean media + cleaned captions:** `remove_fillers("v.mp4", "out/")` then use
  `result.cleaned_media` and `result.cleaned_srt`.
- **Cheap re-tuning of fillers:** run `remove_fillers` once, then re-run with
  `scribe_data=json.load(open("out/scribe.json"))` and a custom `fillers=` set.
- **Chapters from an SRT you already have:** `mixing.detect_chapters(open("v.srt").read())`.
- **Aggressive cut (also drop laughs):** pass `audio_events={"(coughs)", "(laughs)"}`.

## Gotchas

- Needs `ELEVENLABS_API_KEY` (or `ELEVEN_API_KEY`); without it `transcribe`/`srt_for_media`/`remove_fillers`
  raise `RuntimeError`. The disk cache makes repeat runs free and offline.
- `transcribe(...)` defaults to `cache=False`; pass `cache=True` to persist.
- `srt_for_media` won't overwrite an existing `.srt` unless `refresh=True` — it
  assumes the file may be hand-corrected.
- `remove_fillers` re-encodes via ffmpeg (`apply_keeps`); it does NOT use the
  single-`output` protocol — it takes `output_dir=` + `output_media=`.
- `detect_chapters`'s default `segment_fn` needs the `[llm]` extra (`aix`) and an
  LLM call; pass a `segment_fn` to stay dependency-free and offline.
- All times are **seconds** (float) except SRT timestamp strings.
